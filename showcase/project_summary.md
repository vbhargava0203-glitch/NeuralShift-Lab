# NeuralShift Lab

A reproducible empirical investigation of prediction reliability under controlled image shifts. The delivery includes executable software, a real Fashion-MNIST smoke run, and a five-seed reduced CNN study.

In the reduced five-seed pilot, clean accuracy averaged 0.7438 (sample SD 0.0049). MC predictive entropy had mean error-detection AUROC difference -0.0429 relative to MSP, with a paired seed-bootstrap 95% interval [-0.0484, -0.0352], after equally averaging the 21 evaluated conditions within each seed. Thus the proposed MC-entropy advantage was not observed in this pilot. This negative result is conditional on the small CNN, short training, fixed test subset, selected corruptions, and five MC passes; it does not establish a general ranking of uncertainty methods. Comparisons remain exploratory.
