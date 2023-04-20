input="2757matrix.csv"
PATH="YOURPATH"
{
  # readc
  i=1

  # while IFS=',' read -r mid group name rows cols nonzeros
  while IFS=',' read -r mid Group Name rows cols nonzeros
  do
    echo "$mid $Group $Name $rows $cols $nonzeros"
    
    CUDA_VISIBLE_DEVICES=0 ./test ${PATH}/$Group/$Name/$Name.mtx 40000

    i=`expr $i + 1`
  done
} < "$input"

