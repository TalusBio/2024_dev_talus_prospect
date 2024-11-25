# /// script
# dependencies = [
#   "duckdb",
#   "polars",
#   "cloudpathlib[s3]",
#   "tqdm",
#   "pyarrow",
#   "boto3",
# ]
# ///

import json
import shutil
from pathlib import Path

import boto3
import duckdb
import polars as pl
from cloudpathlib import S3Path
from tqdm.auto import tqdm

TESTING_SET = {
    "metadata": "s3://terraform-workstations-bucket/jspaezp/20241022_prospect/TUM_third_pool_meta_data.parquet",
    "files": [
        "s3://terraform-workstations-bucket/jspaezp/20241022_prospect/TUM_third_pool/TUM_third_pool/TUM_third_pool_1_01_01_annotation.parquet",
        "s3://terraform-workstations-bucket/jspaezp/20241022_prospect/TUM_third_pool/TUM_third_pool/TUM_third_pool_2_01_01_annotation.parquet",
        "s3://terraform-workstations-bucket/jspaezp/20241022_prospect/TUM_third_pool/TUM_third_pool/TUM_third_pool_3_01_01_annotation.parquet",
        "s3://terraform-workstations-bucket/jspaezp/20241022_prospect/TUM_third_pool/TUM_third_pool/TUM_third_pool_4_01_01_annotation.parquet",
        "s3://terraform-workstations-bucket/jspaezp/20241022_prospect/TUM_third_pool/TUM_third_pool/TUM_third_pool_5_01_01_annotation.parquet",
        "s3://terraform-workstations-bucket/jspaezp/20241022_prospect/TUM_third_pool/TUM_third_pool/TUM_third_pool_6_01_01_annotation.parquet",
    ],
}

OUTPUT_LOC = "s3://terraform-workstations-bucket/jspaezp/20241115_prospect/"


# con = duckdb.connect(database=":memory:")

# # Read in the metadata file
# meta = con.read_parquet(TESTING_SET["metadata"])
# first_row = meta.fetchone()
# cols = meta.columns
# first_row_dict = {k:v for k,v in zip(cols, first_row)}
# print(first_row_dict)

{
    "raw_file": "01812a_GA3-TUM_third_pool_1_01_01-DDA-1h-R1",
    "scan_number": 23850,
    "modified_sequence": "QLQQIERQLK",
    "precursor_charge": 2,
    "precursor_intensity": 19577780.0,
    "mz": 642.37514,
    "precursor_mz": 642.3751967147814,
    "fragmentation": "CID",
    "mass_analyzer": "ITMS",
    "retention_time": 27.111,
    "indexed_retention_time": 30.483830533968415,
    "andromeda_score": 297.36,
    "peptide_length": 10,
    "orig_collision_energy": 35.0,
    "aligned_collision_energy": 35.0,
}

# # Now the same for the first file
# file = con.read_parquet(TESTING_SET["files"][0])
# first_row = file.fetchone()
# cols = file.columns
# first_row_dict = {k:v for k,v in zip(cols, first_row)}
# print(first_row_dict)

{
    "ion_type": "y",
    "no": 1,
    "charge": 1,
    "experimental_mass": 147.11246,
    "theoretical_mass": 147.112804137,
    "intensity": 0.31,
    "neutral_loss": "",
    "fragment_score": 100,
    "peptide_sequence": "DNYDQLVRIAK",
    "scan_number": 34341,
    "raw_file": "01812a_GA3-TUM_third_pool_1_01_01-DDA-1h-R1",
}


def stage_files(metadata_path, files) -> tuple[Path, list[Path]]:
    # Download the files to a local directory
    s3 = boto3.client("s3")
    partition_name = metadata_path.split("/")[-1].replace("_meta_data.parquet", "")
    local_dir = Path("staged_files") / partition_name

    local_dir.mkdir(exist_ok=True, parents=True)
    for file in tqdm(
        files + [metadata_path],
        desc=f"Downloading files for {partition_name}",
    ):
        filepath = S3Path(file)
        s3.download_file(filepath.bucket, filepath.key, local_dir / filepath.name)

    out_metadata_path = local_dir / S3Path(metadata_path).name
    out_files = [local_dir / S3Path(x).name for x in files]
    return out_metadata_path, out_files


def ingest_to_duckdb(metadata_path, files):
    partition_name = metadata_path.split("/")[-1].replace("_meta_data.parquet", "")
    duckdb_file_name = partition_name + ".duckdb"
    if Path(duckdb_file_name).exists():
        raise FileExistsError(f"File {duckdb_file_name} already exists")

    metadata_path, files = stage_files(metadata_path, files)

    col = {
        "start_nrows_meta": None,
        "start_nrows_files": None,
        "end_nrows_meta": None,
        "end_nrows_files": None,
    }

    scanned_metadata = pl.scan_parquet(metadata_path).with_columns(
        partition=pl.lit(partition_name),
    )
    col["start_nrows_meta"] = scanned_metadata.select(pl.len()).collect().item()

    grouping_cols = [
        "raw_file",
        "modified_sequence",
        "precursor_charge",
        "fragmentation",
        "mass_analyzer",
    ]
    dedup_scanned_metadata = (
        scanned_metadata.group_by(grouping_cols)
        .agg(pl.all().top_k_by(pl.col("andromeda_score"), 2))
        .explode(pl.all().exclude(grouping_cols))
    )
    read_metadata = dedup_scanned_metadata.collect()
    col["end_nrows_meta"] = len(read_metadata)
    with duckdb.connect(database=duckdb_file_name, read_only=False) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS 'precursor' AS SELECT * FROM read_metadata;
            """,
        )

    join_cols_l = ["raw_file", "scan_number", "peptide_sequence"]
    join_cols_r = ["raw_file", "scan_number", "modified_sequence"]

    # I dont have enough mem to use more than 1 file at a time :(
    file_chunksize = 1
    chunked_files = [
        files[i : i + file_chunksize] for i in range(0, len(files), file_chunksize)
    ]
    is_first = True

    for filechunk in tqdm(chunked_files):
        scanned_files = pl.scan_parquet(files)
        col["start_nrows_files"] = scanned_files.select(pl.len()).collect().item()

        scanned_files_j = scanned_files.with_columns(
            partition=pl.lit(partition_name),
        ).join(
            read_metadata.lazy(),
            left_on=join_cols_l,
            right_on=join_cols_r,
            how="semi",
        )

        num_read_lines = 0

        try:
            unique_ion_types = scanned_files.select(
                pl.col("ion_type").unique(),
            ).collect()

            for ion_type in unique_ion_types["ion_type"]:
                filtered_scanned_files = scanned_files_j.filter(
                    pl.col("ion_type") == ion_type,
                )
                read_files = filtered_scanned_files.collect(streaming=True)
                num_read_lines += len(read_files)

                with duckdb.connect(database=duckdb_file_name, read_only=False) as con:
                    if is_first:
                        con.execute(
                            """
                            CREATE TABLE IF NOT EXISTS 'fragment' AS
                            SELECT * FROM read_files;
                            """,
                        )
                        is_first = False
                    else:
                        con.execute(
                            """
                            INSERT INTO fragment 
                            SELECT * FROM read_files;
                            """,
                        )
                    con.execute("CHECKPOINT")

        except pl.exceptions.ComputeError:
            # polars.exceptions.ComputeError: parquet: File out of specification:
            #  underlying IO error: corrupt deflate stream
            with open("corrupt_files.txt", "a") as f:
                for file in filechunk:
                    f.write(file + "\n")
            continue

    shutil.rmtree(metadata_path.parent)

    return duckdb_file_name


def export_to_parquet(duckdb_file_name, partition_name) -> None:
    fg_loc = f"fragments_pq/partition={partition_name}"
    pq_loc = f"precursors_pq/partition={partition_name}"
    Path(fg_loc).mkdir(parents=True)
    Path(pq_loc).mkdir(parents=True)
    with duckdb.connect(database=duckdb_file_name) as con:
        con.execute(
            f"""
            COPY fragment TO '{fg_loc}' (
                FORMAT PARQUET,
                PARTITION_BY (ion_type),
                OVERWRITE TRUE,
                FILENAME_PATTERN 'file_{{i}}'
            )
            """,
        )
        con.execute(
            f"""COPY precursor TO '{pq_loc}' (
                FORMAT PARQUET,
                PARTITION_BY (mass_analyzer, fragmentation),
                OVERWRITE TRUE,
                FILENAME_PATTERN 'file_{{i}}'
            )
            """,
        )


def upload_to_s3(local_dir: Path, s3_path: S3Path, dry_run=False) -> None:
    # Upload a directory.
    # For instance if the local path is "myfiles/mydir/foo.parquet"
    # And I pass as a local dir "myfiles"
    # and the s3 path is "s3://mybucket/somewhere/"
    # the file will be uploaded to "s3://mybucket/somewhere/myfiles/mydir/foo.parquet"
    s3 = boto3.client("s3")
    s3_prefix_key = s3_path.key
    for file in local_dir.rglob("*.parquet"):
        out_key = s3_prefix_key / file.relative_to(local_dir)
        if dry_run:
            pass
        else:
            s3.upload_file(file, s3_path.bucket, out_key)


def main() -> None:
    partitions = json.load(open("data/annot.json"))
    for part_name, part in tqdm(partitions.items()):
        dbname = part_name + ".duckdb"
        try:
            ingest_to_duckdb(part["metadata"], part["files"])
        except FileExistsError:
            pass
        try:
            export_to_parquet(dbname, part_name)
        except FileExistsError:
            pass

    upload_to_s3(
        Path("fragments_pq"),
        S3Path(OUTPUT_LOC) / "fragments",
    )
    upload_to_s3(
        Path("precursors_pq"),
        S3Path(OUTPUT_LOC) / "precursors",
    )


if __name__ == "__main__":
    main()
