# Security Policy

## Supported versions

Only the latest release of arch is supported. This is currently the 9.0.x
series. Security fixes are not backported to earlier releases, so please
upgrade before reporting a problem that you found in an older release.

## Reporting a vulnerability

Please report suspected security vulnerabilities **privately**. Do not report
them in a public issue, pull request or discussion, since that tells everyone
about the problem before a fix is available.

The preferred way to report a vulnerability is GitHub private vulnerability
reporting:

<https://github.com/bashtage/arch/security/advisories/new>

If you cannot use GitHub, email the maintainer, Kevin Sheppard, at
<kevin.k.sheppard@gmail.com>, which is the address listed in `pyproject.toml`.
Use a subject line that makes clear that this is a security report, for example
"arch security vulnerability".

## What to include

A report is easiest to act on if it contains:

- A description of the vulnerability and its impact: what an attacker can do,
  and what they need in order to do it.
- The affected function, class or module, and the version of arch where
  you found the problem.
- Steps to reproduce the problem, ideally as a short, self-contained script.
- The versions of Python, NumPy, SciPy, pandas and statsmodels, and your
  operating system.
- A suggested fix or mitigation, if you have one.
- Whether the problem has been disclosed elsewhere, any disclosure deadline that
  you plan to follow, and whether and how you would like to be credited.

## What to expect

arch is a volunteer project, so responses are best effort. There is no
guaranteed response time or fix date. For a confirmed vulnerability, the
maintainer will work with you on a fix and on the timing of its disclosure.
