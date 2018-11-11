for /L %%i in (115640,2000,115640) do (
    ..\..\Build\x64\Release\caffe.exe test -iterations=70 -model=./net.prototxt -weights=./snapshot/snapshot_iter_%%i.caffemodel -gpu=0
    )
pause