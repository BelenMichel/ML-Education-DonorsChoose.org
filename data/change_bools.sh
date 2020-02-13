for INDEX in `seq 1 3`; do
  for SET in "test" "train"; do 
    FILE=projects_2012_2013_${SET}_${INDEX}.csv
    sed 's/False/0/g' $FILE | sed 's/True/1/g' > ${FILE}_;
    mv ${FILE}_ ${FILE}
  done; 
done
