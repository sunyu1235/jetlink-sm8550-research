package io.zoompilot.jetlink.server;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import org.json.JSONObject;

/** Line-oriented console for the installed Jetlink native library. */
public final class Native {
    private Native() {}

    public static native String start(String config);
    public static native void stop();
    public static native String command(String json);
    public static native String snapshot(long after, int timeoutMs);
    public static native String logs(long after);

    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("Usage: Native <nativeLibDir> <cacheDir>");
            System.exit(2);
            return;
        }

        boolean loaded = false;
        int exitCode = 0;
        try {
            System.load(new File(args[0], "libjetlink.so").getAbsolutePath());
            loaded = true;
            JSONObject config = new JSONObject();
            config.put("cache", args[1]);
            config.put("backend", "litert");
            config.put("device", "gpu");
            config.put("chip", "SM8550");
            config.put("native_library_dir", args[0]);
            config.put("preload", false);
            config.put("listen", false);
            config.put("port", 0);
            config.put("usb", false);
            config.put("keep_alive", false);
            config.put("keep_cpu_warm", false);

            String error = start(config.toString());
            if (error != null) {
                throw new IllegalStateException("Jetlink start failed: " + error);
            }
            System.out.println("READY");
            System.out.flush();

            BufferedReader input = new BufferedReader(
                    new InputStreamReader(System.in, StandardCharsets.UTF_8));
            String line;
            while ((line = input.readLine()) != null) {
                if ("@quit".equals(line)) {
                    break;
                }
                String reply;
                if ("@start-snapshot".equals(line)) {
                    reply = snapshot(0L, 1000);
                } else if ("@logs".equals(line)) {
                    reply = logs(0L);
                } else {
                    reply = command(line);
                }
                System.out.println(reply);
                System.out.flush();
            }
        } catch (Throwable error) {
            error.printStackTrace(System.err);
            exitCode = 1;
        } finally {
            if (loaded) {
                try {
                    stop();
                } catch (Throwable error) {
                    error.printStackTrace(System.err);
                    exitCode = 1;
                }
            }
        }
        if (exitCode != 0) {
            System.exit(exitCode);
        }
    }
}
