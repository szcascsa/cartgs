package ai.mycartgs.quest;

// A package ClassLoader is required for Android's declared vendor-native library namespace.
// All rendering, tracking, UI and network execution remain in C++ NativeActivity.
public final class QuestActivity extends android.app.NativeActivity {
    @Override
    protected void onCreate(android.os.Bundle state) {
        super.onCreate(state);
        java.io.File base = getExternalFilesDir(null);
        if (base != null) {
            java.io.File reports = new java.io.File(base, "benchmarks");
            if (!reports.exists()) reports.mkdirs();
            reports.setReadable(true, false);
            reports.setExecutable(true, false);
            // Scoped storage can block adb pull even for external app reports.
            // Explicit export reads only the newest two small CPU CSVs, never models.
            if (getIntent().getBooleanExtra("mycartgs.export_cpu_reports", false)) {
                java.io.File[] files = reports.listFiles((dir, name) -> name.matches("cpu_inference_[0-9]+\\.csv"));
                if (files != null) {
                    java.util.Arrays.sort(files, java.util.Comparator.comparing(java.io.File::getName));
                    for (int i = Math.max(0, files.length - 2); i < files.length; ++i) {
                        try {
                            if (files[i].length() > 1024 * 1024) throw new java.io.IOException("CPU CSV too large");
                            byte[] bytes = java.nio.file.Files.readAllBytes(files[i].toPath());
                            String data = android.util.Base64.encodeToString(bytes, android.util.Base64.NO_WRAP);
                            StringBuilder digest = new StringBuilder();
                            for (byte b : java.security.MessageDigest.getInstance("SHA-256").digest(bytes))
                                digest.append(String.format(java.util.Locale.ROOT, "%02x", b & 255));
                            android.util.Log.i("SplatKitCpuCsv", "BEGIN " + files[i].getName() + " " + bytes.length + " " + digest);
                            for (int p = 0; p < data.length(); p += 2048)
                                android.util.Log.i("SplatKitCpuCsv", "DATA " + data.substring(p, Math.min(p + 2048, data.length())));
                            android.util.Log.i("SplatKitCpuCsv", "END " + files[i].getName());
                        } catch (Exception error) { android.util.Log.e("SplatKitCpuCsv", "EXPORT_FAILED " + files[i].getName(), error); }
                    }
                }
            }
        }
    }
}
