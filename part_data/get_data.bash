#!/bin/bash

set -x
set -e
set -u

mkdir -p fragments_pq
mkdir -p precursors_pq

aws s3 cp --recursive "s3://terraform-workstations-bucket/jspaezp/20241115_prospect_hive/fragments_pq/partition=Kmod_GlyGly" "fragments_pq/partition=Kmod_GlyGly"
aws s3 cp --recursive "s3://terraform-workstations-bucket/jspaezp/20241115_prospect_hive/precursors_pq/partition=Kmod_GlyGly" "precursors_pq/partition=Kmod_GlyGly"
