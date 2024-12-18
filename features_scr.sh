
#!/bin/bash

# The directory that contains the folders to be renamed
DIR="/home/mani/Desktop/hyperGNN/goods"

# The name of the folders to be renamed
NAME="features"

# A counter to keep track of the number of folders
COUNTER=0

cd $DIR
# $RESULT_PATH/fix107/${archs[i]}/$cw/$ur/$bu/${circuit_titan[value]}/features/vpr.txt

for folder in ./*/*/*/*/*/*/; do
    echo $folder
    mv $DIR/$folder/features $DIR/$folder/"features ($COUNTER)"
    COUNTER=$((COUNTER+1))

done

# # Print a message when done
# echo "Renamed $((COUNTER-1)) folders in $DIR."