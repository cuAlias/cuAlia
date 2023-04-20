input="2757matrix.csv"
PATH="YOURPATH"

{
  # readc
  i=1

  # while IFS=',' read -r mid group name rows cols nonzeros
  while IFS=',' read -r mid Group Name rows cols nonzeros
  do
    echo "$mid $Group $Name $rows $cols $nonzeros"
    
    CUDA_VISIBLE_DEVICES=0 ./sage ${PATH}/$Group/$Name/$Name.mtx 40000 # run in dev0 4090
    CUDA_VISIBLE_DEVICES=1 ./sage ${PATH}/$Group/$Name/$Name.mtx 40000 # run in dev1 4070T
    
    # CUDA_VISIBLE_DEVICES=0 ./deepwalk ${PATH}/$Group/$Name/$Name.mtx 40000 # run in dev0 4090
    # CUDA_VISIBLE_DEVICES=1 ./deepwalk ${PATH}/$Group/$Name/$Name.mtx 40000 # run in dev1 4070T
    
    i=`expr $i + 1`
  done
} < "$input"

