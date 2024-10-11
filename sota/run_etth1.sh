START=1 STEP=1 LAMBDA=0.5 python sota_ettm2.py  # model without pred self-supervision

donemodels=`tail -n 1 sota.csv`
if [[ "$donemodels" = "" ]]; then
  initial=0.4
  inc=0.03
  star=0.9
  for s in $(LC_ALL=en_US.UTF-8 seq $initial $inc $star); 
  do
    initial2=0.05
    inc2=0.06
    end2=0.8
    for w in $(LC_ALL=en_US.UTF-8 seq $initial2 $inc2 $end2);
    do
      START=$s STEP=$w LAMBDA=0.5 python sota_ettm2.py
    done
  done

else
  IFS=',' read -ra ADDR <<< "$donemodels"

  initial2=${ADDR[1]}
  inc2=0.06
  end2=0.8

  s=${ADDR[0]}
  
  for w in $(LC_ALL=en_US.UTF-8 seq $initial2 $inc2 $end2);
    do
      START=$s STEP=$w LAMBDA=0.5 python sota_ettm2.py
    done
  
  inc=0.03
  initial="${ADDR[0]} + $inc" | bc
  star=0.9
 
  for s in $(LC_ALL=en_US.UTF-8 seq $initial $inc $star); 
  do
    initial2=0.05
    inc2=0.06
    end2=0.8
    for w in $(LC_ALL=en_US.UTF-8 seq $initial2 $inc2 $end2);
    do
      START=$s STEP=$w LAMBDA=0.5 python sota_ettm2.py
    done
  done

fi


