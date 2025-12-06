"""
Script to initialize Great Expectations and create an Expectation Suite
"""
import great_expectations as gx
import pandas as pd
import os

def main():
    # We will use the filesystem context to persist the suite
    if not os.path.exists("gx"):
        os.makedirs("gx")

    context = gx.get_context(project_root_dir=".")

    suite_name = "data_suite"

    # Check if suite exists
    try:
        suite = context.suites.get(suite_name)
        print(f"Loaded existing suite: {suite_name}")
        # If it exists, we might want to delete it to start fresh, or just use it.
        context.suites.delete(suite_name)
        print(f"Deleted existing suite: {suite_name}")
        suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
        print(f"Created new suite: {suite_name}")
    except:
        suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
        print(f"Created new suite: {suite_name}")

    # Define Data Source
    datasource_name = "my_pandas_datasource"
    try:
        datasource = context.data_sources.get(datasource_name)
    except:
        datasource = context.data_sources.add_pandas(datasource_name)

    asset_name = "raw_dataset"
    try:
        asset = datasource.get_asset(asset_name)
    except:
        asset = datasource.add_csv_asset(name=asset_name, filepath_or_buffer="data/raw/dataset.csv")

    try:
        batch_definition = asset.get_batch_definition("whole_dataframe")
    except KeyError:
        batch_definition = asset.add_batch_definition_whole_dataframe("whole_dataframe")

    batch_request = batch_definition.build_batch_request()

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )

    # Define Expectations
    validator.expect_table_columns_to_match_ordered_list(
        column_list=['feature1', 'feature2', 'feature3', 'feature4', 'target']
    )

    # Feature 1
    validator.expect_column_values_to_not_be_null("feature1")
    validator.expect_column_values_to_be_between("feature1", min_value=0, max_value=20)

    # Feature 2
    validator.expect_column_values_to_not_be_null("feature2")
    validator.expect_column_values_to_be_between("feature2", min_value=0, max_value=20)

    # Target
    validator.expect_column_values_to_be_in_set("target", [0, 1])

    # Update the suite in the context store
    # validator.expectation_suite is the object, we need to save it via context or validator.
    # validator.save_expectation_suite() was failing because it tried to 'add'.

    # Since GX complains about freshness, let's try to fetch the suite from the context again and make sure it is saved.

    # validator.expectation_suite is modified in memory.
    # To update it in the store:
    context.suites.add_or_update(validator.expectation_suite)
    print(f"Expectation suite '{suite_name}' saved.")

    # Re-fetch suite to ensure it is 'fresh' for the validation definition
    suite = context.suites.get(suite_name)

    # Create a Validation Definition (GX 1.0+ Requirement)
    validation_definition_name = "data_quality_validation_def"

    try:
        validation_definition = context.validation_definitions.get(validation_definition_name)
        # Update it? Or delete/recreate
        context.validation_definitions.delete(validation_definition_name)
    except:
        pass

    validation_definition = context.validation_definitions.add(
        gx.core.validation_definition.ValidationDefinition(
            name=validation_definition_name,
            data=batch_definition,
            suite=suite
        )
    )

    # Create a Checkpoint
    checkpoint_name = "data_quality_checkpoint"

    # Check if checkpoint exists
    try:
        context.checkpoints.delete(checkpoint_name)
    except:
        pass

    # In GX 1.0+, Checkpoints use validation_definitions
    checkpoint = context.checkpoints.add(
        gx.checkpoint.Checkpoint(
            name=checkpoint_name,
            validation_definitions=[validation_definition],
            result_format="COMPLETE"
        )
    )

    print(f"Checkpoint '{checkpoint_name}' saved.")

    print("Initialization complete.")

if __name__ == "__main__":
    main()
