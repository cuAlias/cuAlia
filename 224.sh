input="2757matrix.csv"

{
  # readc
  i=1

  # while IFS=',' read -r mid group name rows cols nonzeros
  while IFS=',' read -r mid Group Name rows cols nonzeros
  do
    echo "$mid $Group $Name $rows $cols $nonzeros"

    #./test 2 160 /home/fx/Desktop/MM/$Group/$Name/$Name.mtx
    #./test /media/hemeng/2TB/MM/$Group/$Name/$Name.mtx `expr $nonzeros / 20`# 5%
    # ./test /media/hemeng/2TB/MM/$Group/$Name/$Name.mtx `expr $nonzeros / 10`# 10%
    # ./test /media/hemeng/2TB/MM/$Group/$Name/$Name.mtx `expr \( $nonzeros \* 15 \) / 100`# 15%
    CUDA_VISIBLE_DEVICES=0 ./test /media/hemeng/2TB/MM/$Group/$Name/$Name.mtx 40000
    # CUDA_VISIBLE_DEVICES=1 ./test /media/hemeng/2TB/MM/$Group/$Name/$Name.mtx 40000
     #./test /media/hemeng/2TB/MM/$Group/$Name/$Name.mtx `expr $nonzeros / 5`# 20%
    # ./test /media/hemeng/2TB/MM/$Group/$Name/$Name.mtx `expr $nonzeros / 4`# 25%
    i=`expr $i + 1`
  done
} < "$input"

