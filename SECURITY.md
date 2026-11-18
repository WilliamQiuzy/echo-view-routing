# Security

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private reporting instead: the "Report a vulnerability" button under the repository's Security tab. You can expect a first answer within a week.

## What is in scope

- **Loading weights.** The package loads `.safetensors` files, and `.pt` files with `torch.load(..., weights_only=True)`. Neither runs code from the file. Before it calls a system the reference, it checks every weight file against the SHA-256 that ships with the code. Report anything that lets a weight file run code or pass that check when it should not.
- **Reading recordings.** DICOM files, video files and images come from outside. Report any input that makes a reader run code, write outside the output paths, or use unbounded memory.

## Patient data

A leak of patient data is a security problem. If you find a recording, a frame or an identifier of a restricted dataset in this repository, in its history or in the weights repository, report it privately in the same way.

## Not a medical device

This software is for research. It is not validated for clinical use, and a wrong answer of the model is not a vulnerability.
