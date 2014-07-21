**Source installers**

```
python setup.py sdist --formats=bztar,zip upload -r pypi
```

**Building Wheels**

```
python setup.py bdist_wheel upload -r pypi
```

**Executalble installers**

```
python setup.py bdist_wininst upload -r pypi
```
