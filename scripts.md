## Check types with mypy

```bash
mypy ./src
```

## Run project

```bash
python3 src/main-1-tree.py
python3 src/main-2-ensamble.py
```

## Generate report

```bash
pandoc README.md -o out/report.pdf --pdf-engine=xelatex
```

## Git commands

```bash  
git add -u; git commit --amend --no-edit; git push --force-with-lease
```