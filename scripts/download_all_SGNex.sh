aws s3 ls s3://sg-nex-data/data/processed_data/m6Anet/ --no-sign-request | awk '{print $2}' | sed 's#/$##' > ./data_list.txt
LINK_VAR="s3://sg-nex-data/data/processed_data/m6Anet"
export LINK_VAR
cat data_list.txt | xargs -n 1 -P 4 -I {} bash -c '
  VAR_NAME={}
  echo "Processing $VAR_NAME ..."
  aws s3 cp "${LINK_VAR}/${VAR_NAME}/data.json" - --no-sign-request |
  gzip > "../data/json/${VAR_NAME}.json.gz" &&
  echo "Converting $VAR_NAME ..." &&
  bash JsonToCsv_v3.sh "../data/json/${VAR_NAME}.json.gz" "../data/csv/${VAR_NAME}.csv"'