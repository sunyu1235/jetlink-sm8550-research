# Completed native recorded-road gates

The native compact QNN upstream and three-tail contexts, with the remainder of the full recurrent model on CPU, pass both recorded sequences:

- [Night320](night320/summary.json): all320 and the predeclared final160 after160 warmup pass.
- [Day320](day320/summary.json): all320 and the predeclared final160 after160 warmup pass.

The original checker and its thresholds are unchanged. Teacher and candidate propagate independent states; no acceptance outputs were fitted. All model, bridge, package and context identities in the checked artifact manifest remained unchanged. The day run recovered from its own durable frame-300 checkpoint and regenerated the final20 frames after an environment interruption. Earlier completed night and prefix gates were reused, not claimed as newly rerun.

[final-qualification.json](final-qualification.json) is copied byte-identically from the completed combined result. Both folders retain the complete original checker logs. The original input/reference checks are in `../../road-replay/road-recovery-provenance.json`. The earlier `qualification-snapshot.json` is preserved as historical evidence from before day320 completed and is superseded by the final qualification.

This is offline host numerical agreement on two reconstructed recorded sequences, not phone execution, ground-truth driving accuracy, driving safety, or a 32ms result. Phone numerical and timing gates remain unverified. The added evidence excludes raw inputs/outputs, model weights, SDK material and binaries.
