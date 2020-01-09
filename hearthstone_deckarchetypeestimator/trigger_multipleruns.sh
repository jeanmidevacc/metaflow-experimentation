NUMBERRUNS=$1
FILE=$2
echo $NUMBERRUNS $FILE
cd /home/ubuntu/development/metaflow-experimentation/hearthstone_deckarchetypeestimator
pwd
counter=1
while [ $counter -le $NUMBERRUNS ]
do
    python $FILE --environment=conda run
    ((counter++))
done

