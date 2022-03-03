# Compilation
```bash
mkdir build && cd build
cmake  ..
make
```

# Execution

Arguments:
- \<Image : filename\>
- \<Filter : filename\>
- \<Filter type : 'b' for Blur | 'ed' for Edge detector\>
- \<Filter index : integer index for select one filter or 'all' for select all\>
- \<High threshold : decimal number\> (only for 'ed' filter)
- \<Low threshold : decimal number\> (only for 'ed' filter)

Exemples :
```
build/TP1
build/TP1 ../data/img/ville.jpg ../data/filter/Prewitt_4D.txt ed all 0.2 0.9
build/TP1 ../data/img/ville.jpg ../data/filter/Kirsch_2D.txt ed all 0.2 0.9
build/TP1 ../data/img/ville.jpg ../data/filter/Blur3.txt ed all
build/TP1 ../data/img/ville.jpg ../data/filter/GaussianBlur3.txt ed 0
```