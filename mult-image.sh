#!/bin/bash

ITER=16

for (( i=16; i<=348; i++ ))
do
    echo "执行第 $ITER 次 make terminal..."
    make terminal ITER=$ITER
    wait
    
    # 增加执行计数
    ITER=$((ITER+1))

    # (可选) 如果想在每次执行间增加延时, 取消下行的注释
    # sleep 5
done