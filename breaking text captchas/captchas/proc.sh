#!/bin/bash

files=$(ls)
echo $files | tr ' ' '\n' > files.txt

itr=0
input_file="files.txt"
while IFS= read -r line
do
  if [ $itr -eq 9 ]
  then
    mv "${line}" captchas/"${line}"
    itr=0
  else
    itr=$((itr+1))
  fi
done < $input_file
