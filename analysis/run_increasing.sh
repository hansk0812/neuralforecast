START=1 STEP=1 LAMBDA=0.5 python model_increasing.py # model without pred self-supervision
initial=0.05
inc=0.03
star=0.7
end=1
for s in $(LC_ALL=en_US.UTF-8 seq $initial $inc $star); 
do
  initial2=0.05
  inc2=0.03
  end2=0.4
  for w in $(LC_ALL=en_US.UTF-8 seq $initial2 $inc2 $end2);
  do
    START=$s STEP=$w LAMBDA=0.5 python model_increasing.py
  done
done
