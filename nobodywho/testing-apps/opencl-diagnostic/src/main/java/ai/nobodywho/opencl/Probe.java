package ai.nobodywho.opencl;

public final class Probe {
    static { System.loadLibrary("opencl_probe"); }
    public static native boolean run();
}
