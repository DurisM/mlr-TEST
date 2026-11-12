# mlr TEST — Exercise 8, Task 4

Controlled leaf-classifier comparisons requested for this separate public test repository.

The code keeps Task 3's `LeafDataset`, `LeafNetBN` and batch training loop. The only
experiments are colour augmentation; pretrained MobileNet with its backbone frozen
versus fully fine-tuned; and augmentation, cosine scheduling and longer training.
There are 11 setups, each measured with seeds 0, 1 and 2 on the same group split.

The selected setup is frozen MobileNet, final layer trained for 15 epochs:
**95.50% clean / 85.59% camera-like accuracy** averaged over three seeds.
The exported seed-0 checkpoint itself scored 94.59% / 81.08%.

- [Report draft](REPORT.md): measured results and interpretation; the student must
  revise it in their own words and add real phone-camera observations.
- [Code](task4.py) and [notebook](Copy_of_ex08_student.ipynb).
- [Per-seed results](task4_results/results.csv) and
  [paired comparisons](task4_results/comparison_table.md).
- [Selected ONNX model](task4_results/phone/leaf_model.onnx),
  [metadata](task4_results/phone/model_info.json), and
  [export checks](task4_results/phone/export_validation.json).

Use Python 3.11 and install `task4_requirements.txt`. Download/extract the supplied
group ZIP into `data/leaves` using Task 2a of the notebook, then run:

```sh
python task4.py compare
python task4.py combine
python task4.py export
```

The saved combination plan records decisions after the initial 27 runs. Every row
uses the same 136 training and 37 validation images. Camera-like validation is
generated once with seed 2026. All rows use final-epoch weights. The selection rule
is highest mean camera-like accuracy, then clean accuracy and smaller model size.
Export seed 0 was fixed in advance. Raw images, transformed validation tensors,
training checkpoints and dependency caches are not committed; the code regenerates them.

The ONNX export matched PyTorch on all 37 clean and 37 camera-like inputs, with a
maximum probability error of 8.64e-6. The
[course camera app](https://tomasvicar.github.io/MLR-public/apps/classifier/) can load
the ONNX file. The browser upload was blocked by declined file-upload permission;
no successful browser inference or physical phone-camera trial is claimed.
