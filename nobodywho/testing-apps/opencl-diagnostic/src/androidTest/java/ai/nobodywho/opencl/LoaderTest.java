package ai.nobodywho.opencl;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.assertTrue;

@RunWith(AndroidJUnit4.class)
public class LoaderTest {
    @Test public void createAndUseSubBuffers() {
        assertTrue("OpenCL probe failed; see OpenCLProbe in logcat", Probe.run());
    }
}
