pub mod event;
pub mod response;

use std::collections::HashMap;

use encoding_rs::Decoder;
use llama_cpp_2::{model::LlamaModel, token::LlamaToken};
use tracing::trace;

use crate::{
    event_stream::{
        event::{
            EventKind, OutputIndex, OutputItemAddedEvent, OutputItemDoneEvent, SequenceNumber,
            StreamEvent,
        },
        response::{FunctionCallId, Item, ItemId, ItemType, ResponseId, ResponseObject, Status},
    },
    tool_calling::ToolFormat,
};

struct EventConsumer {
    responses: HashMap<ResponseId, ResponseObject>,
    active_response_id: Option<ResponseId>,
    cur_sequence_number: SequenceNumber,
}

impl EventConsumer {
    pub fn new() -> EventConsumer {
        EventConsumer {
            responses: HashMap::new(),
            active_response_id: None,
            cur_sequence_number: SequenceNumber::start(),
        }
    }

    pub fn consume_event(&mut self, event: StreamEvent) -> Result<(), EventStreamError> {
        match event {
            StreamEvent {
                kind: EventKind::Created { response },
                sequence_number,
                ..
            } => {
                assert_eq!(
                    sequence_number,
                    SequenceNumber::start(),
                    "Created event must have sequence index 0"
                );
                if self.active_response_id.is_some() {
                    panic!(
                        "Created event received, but there is already an active response with id {:?}",
                        self.active_response_id
                    );
                }
                // Sequence indices restart with each response.
                self.cur_sequence_number = sequence_number.next();
                let response_id = response.id().clone();
                self.responses.insert(response_id.clone(), response);
                self.active_response_id = Some(response_id);
            }
            event => {
                if event.sequence_number != self.cur_sequence_number {
                    panic!(
                        "Event sequence index mismatch: expected {:?}, got {:?}",
                        self.cur_sequence_number, event.sequence_number
                    );
                }
                self.cur_sequence_number = event.sequence_number.next();

                let active_response_id = self
                    .active_response_id
                    .as_ref()
                    .expect("Event received, but there is no active response to consume it for");
                let response = self.responses.get_mut(active_response_id).expect(
                    "Active response id is set, but the response does not exist in the responses map",
                );
                response.consume_event(event)?;

                if response.status() == Status::Completed {
                    self.active_response_id = None;
                }
            }
        }
        Ok(())
    }
}

pub enum EventStreamError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventStreamState {
    /// Writing a message to the user.
    Message,
    ToolCall,
    Thinking,
    /// Skipping white space after a tool call or thinking and before message to the user.
    WhiteSpace,
    /// Seen the end of generation token. Calling `consume_token` after this will return an error.
    Done,
}

pub struct EventStream<'a, R: rand::Rng> {
    // configuration
    model: &'a LlamaModel,
    tool_format: Option<ToolFormat>,
    // state
    state: EventStreamState,
    token_buffer: Vec<LlamaToken>,
    str_buffer: String,
    decoder: Decoder,
    rng: R,
    // the response object being built up from events
    response: ResponseObject,
    history: Vec<StreamEvent>,
}

impl<'a, R: rand::Rng> EventStream<'a, R> {
    pub fn new(
        model: &'a LlamaModel,
        tool_format: Option<ToolFormat>,
        start_thinking: bool,
        rng: R,
    ) -> (Self, Vec<StreamEvent>) {
        let mut stream = EventStream {
            model,
            tool_format,
            state: if start_thinking {
                EventStreamState::Thinking
            } else {
                EventStreamState::Message
            },
            token_buffer: Vec::new(),
            str_buffer: String::new(),
            decoder: encoding_rs::UTF_8.new_decoder(),
            rng,
            response: ResponseObject::init(ResponseId("PLACEHOLDER".to_string())),
            history: Vec::new(),
        };

        let events = vec![
            stream.new_event(EventKind::Created {
                response: stream.response.clone(),
            }),
            stream.new_event(EventKind::InProgress {
                response: stream.response.clone(),
            }),
        ];

        (stream, events)
    }

    fn new_event(&mut self, kind: EventKind) -> StreamEvent {
        let seq = SequenceNumber(self.history.len());
        let event = StreamEvent {
            sequence_number: seq,
            kind,
        };
        self.history.push(event.clone());
        event
    }

    fn token_to_string(decoder: &mut Decoder, model: &LlamaModel, token: LlamaToken) -> String {
        // Attempt to convert token(s) to bytes
        let token_bytes = match model.token_to_piece_bytes(token, 64, true, None) {
            Err(llama_cpp_2::TokenToStringError::InsufficientBufferSpace(i)) => model
                .token_to_piece_bytes(
                    token,
                    (-i).try_into().expect("Error buffer size is positive"),
                    true,
                    None,
                ),
            x => x,
        }
        .unwrap();

        // Attempt to convert bytes to utf8 string.
        let max_len = decoder
            .max_utf8_buffer_length(token_bytes.len())
            .expect("max buffer length should never exceed a usize");
        let mut token_str = String::with_capacity(max_len);

        // this is where the utf-8 decoder handles partial unicode
        // it'll write whatever printable chars it can into `token_str`
        // and retain partial codepoints for next decoding attempt
        let (_result, _bytes_read, _had_errors) =
            decoder.decode_to_string(&token_bytes, &mut token_str, false);
        token_str
    }

    pub fn consume_token(
        &mut self,
        token: LlamaToken,
    ) -> Result<Option<StreamEvent>, EventStreamError> {
        self.token_buffer.push(token);

        let token_str = Self::token_to_string(&mut self.decoder, self.model, token);

        self.str_buffer.push_str(&token_str);

        let has_eog = self.model.is_eog_token(token);
        trace!(?token, ?token_str, ?has_eog);

        self.str_buffer.push_str(&token_str);

        self.parse_event_from_buffer()
    }

    fn in_progress_item_index(&self) -> Option<OutputIndex> {
        self.response
            .output()
            .iter()
            .enumerate()
            .find_map(|(i, item)| {
                if item.status == Status::InProgress {
                    Some(OutputIndex(i))
                } else {
                    None
                }
            })
    }

    fn next_output_index(&self) -> OutputIndex {
        OutputIndex(self.response.output().len())
    }

    fn parse_event_from_buffer(&mut self) -> Result<Option<StreamEvent>, EventStreamError> {
        if self.str_buffer.is_empty() {
            debug_assert!(self.token_buffer.is_empty());
            return Ok(None);
        }

        if let Some(tool_format) = &self.tool_format {
            if self.str_buffer == tool_format.begin_token() {
                if self.in_progress_item_index().is_some() {
                    panic!("Received begin token, but there is already an in-progress item");
                }
                let output_index = self.next_output_index();
                let item_id = ItemId::generate_function_call(&mut self.rng);
                let call_id = FunctionCallId::generate(&mut self.rng);
                let item = Item::init_function_call(item_id, call_id);
                let event = self.new_event(EventKind::OutputItemAdded(OutputItemAddedEvent {
                    output_index,
                    item,
                }));
                self.response.consume_event(event.clone())?;
                return Ok(Some(event));
            }
            if self.str_buffer == tool_format.end_token() {
                if let Some(output_index) = self.in_progress_item_index() {
                    let mut item = self.response.output()[output_index.0].clone();
                    if item.kind.item_type() != ItemType::FunctionCall {
                        panic!("Received end function call token, but the in-progress item is not a function call");
                    }
                    item.status = Status::Completed;
                    let event = self.new_event(EventKind::OutputItemDone(OutputItemDoneEvent {
                        output_index,
                        item,
                    }));
                    self.response.consume_event(event.clone())?;
                    return Ok(Some(event));
                } else {
                    panic!("Received end function call token, but there is no in-progress item");
                }
            }
        }
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Replay a recorded conversation through an `EventConsumer`.
    ///
    /// The asserts inside `consume_event` are the assertions; this only has to get
    /// every event in, in order.
    fn replay(recording: &str) {
        let events: Vec<StreamEvent> = serde_json::from_str(recording)
            .expect("every event in the recording must map onto an `EventKind`");

        let mut consumer = EventConsumer::new();
        for event in events {
            if consumer.consume_event(event).is_err() {
                panic!("EventConsumer rejected an event");
            }
        }
    }

    /// One test per recorded conversation, for the recordings in one directory.
    macro_rules! conversations {
        ($dir:literal) => {
            #[test]
            fn simple_text() {
                replay(include_str!(concat!($dir, "simple_text.json")));
            }

            #[test]
            fn multi_turn_text() {
                replay(include_str!(concat!($dir, "multi_turn_text.json")));
            }

            #[test]
            fn single_tool_call() {
                replay(include_str!(concat!($dir, "single_tool_call.json")));
            }

            #[test]
            fn parallel_tool_calls() {
                replay(include_str!(concat!($dir, "parallel_tool_calls.json")));
            }

            #[test]
            fn reasoning_with_tool_call() {
                replay(include_str!(concat!($dir, "reasoning_with_tool_call.json")));
            }

            #[test]
            fn incomplete_max_output_tokens() {
                replay(include_str!(concat!(
                    $dir,
                    "incomplete_max_output_tokens.json"
                )));
            }
        };
    }

    mod openai_gpt_5_nano {
        use super::*;
        conversations!("../../../agatest/streams/");
    }

    mod openrouter_gpt_5_nano {
        use super::*;
        conversations!("../../../agatest/streams/openrouter/gpt-5-nano/");
    }

    mod openrouter_claude_haiku_4_5 {
        use super::*;
        conversations!("../../../agatest/streams/openrouter/claude-haiku-4.5/");
    }
}
