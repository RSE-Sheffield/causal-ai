# Security Policy

## Supported versions

Only the latest release of `causal-ai` receives security updates. As this is an early-stage research tool (v0.1.x), we do not maintain patches for older versions.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a vulnerability

If you find a security vulnerability in this repository, please do not open a public GitHub issue. Instead, report it directly to the team by emailing [rse@sheffield.ac.uk](mailto:rse@sheffield.ac.uk) with a description of the issue and steps to reproduce it.

We will aim to acknowledge your report within 5 working days and provide an update on how we plan to address it. If the vulnerability is confirmed, we will work on a fix and release a patch as soon as reasonably possible, crediting the reporter if they wish.

Note that `causal-ai` is a research tool and does not handle sensitive user data or authentication. The main security considerations are around dependency vulnerabilities and the integrity of HPC job outputs.
