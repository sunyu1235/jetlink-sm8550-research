# Recorded input reconstruction recipes

The `night/` and `day/` folders contain original preparation scripts, preprocessing notes, public source URLs, frozen-input hashes and selected validation evidence. Videos, rlogs, raw telemetry metadata, images, arrays, source dependencies and model weights are deliberately not bundled.

Start with each `PREPROCESSING.md`, `source_manifest.json`, and `provenance.json`. These retain the precise upstream references, source file hashes, geometry assumptions, reconstruction caveats and state reset/warmup limitations. Workspace paths identify the original run and must be adapted for a new checkout. Obtain upstream dependencies with their existing licenses intact. Do not substitute current upstream code for pinned references without a new validation.

The completed tests compare numerical output against the original teacher on reconstructed public recorded-camera inputs. They are not ground-truth driving accuracy or safety tests. No raw logs or personally collected user data are published by this snapshot.
