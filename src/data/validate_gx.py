"""
Data Validation Module using Great Expectations
"""
import great_expectations as gx
import sys
import os
import json
import webbrowser

def validate_data(data_path="data/raw/dataset.csv", run_name="manual_run"):
    print(f"\n{'='*50}")
    print("Starting Data Validation with Great Expectations")
    print(f"{'='*50}\n")

    # 1. Load Context
    context = gx.get_context(project_root_dir=".")

    # 2. Setup Validation

    print(f"Validating file: {data_path}")

    datasource_name = "my_pandas_datasource"
    datasource = context.data_sources.get(datasource_name)

    asset_name = "dynamic_asset"
    try:
        datasource.delete_asset(asset_name)
    except:
        pass

    asset = datasource.add_csv_asset(name=asset_name, filepath_or_buffer=data_path)
    batch_definition = asset.add_batch_definition_whole_dataframe("whole_dataframe")

    suite_name = "data_suite"
    suite = context.suites.get(suite_name)

    val_def_name = "dynamic_validation_def"
    try:
        context.validation_definitions.delete(val_def_name)
    except:
        pass

    validation_definition = context.validation_definitions.add(
        gx.core.validation_definition.ValidationDefinition(
            name=val_def_name,
            data=batch_definition,
            suite=suite
        )
    )

    # Create an ephemeral Checkpoint or update existing one
    temp_checkpoint_name = "temp_checkpoint"
    try:
        context.checkpoints.delete(temp_checkpoint_name)
    except:
        pass

    checkpoint = context.checkpoints.add(
        gx.checkpoint.Checkpoint(
            name=temp_checkpoint_name,
            validation_definitions=[validation_definition],
            result_format="COMPLETE"
        )
    )

    result = checkpoint.run(
        run_id=gx.core.run_identifier.RunIdentifier(run_name=run_name)
    )

    # 3. Analyze Results
    success = result.success

    print(f"\nValidation Success: {success}")

    # Check for failures
    if not success:
        print("\n❌ VALIDATION FAILED")
        for validation_result_identifier, validation_result in result.run_results.items():
            for exception in validation_result.results:
                if not exception.success:
                    print(f"  • {exception.expectation_config.type}: Failed")
                    print(f"    Kwargs: {exception.expectation_config.kwargs}")
                    print(f"    Result: {exception.result}\n")

        # Alerting
        send_alert(f"Data Validation Failed for {data_path}!", result)
    else:
        print("\n✓ VALIDATION PASSED")
        # Print basic stats from the first validation result
        first_result = list(result.run_results.values())[0]
        print(f"Statistics: {first_result.statistics}")

    # Build Data Docs
    print("\nBuilding Data Docs...")
    context.build_data_docs()

    # Get path to docs
    try:
        docs_url = context.get_docs_sites_urls()[0]['site_url']
        print(f"Data Docs generated at: {docs_url}")
    except:
        print("Data Docs generated (could not retrieve URL). check gx/uncommitted/data_docs/")

    # return success status
    return success

def send_alert(message, result):
    """
    Mock alerting function
    In production, this would send to Slack/Email/PagerDuty
    """
    print(f"\n{'!'*50}")
    print(f"ALERT TRIGGERED: {message}")
    print(f"Run ID: {result.run_id}")
    print("Sending notification to #data-alerts...")
    print(f"{'!'*50}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "data/raw/dataset.csv"

    success = validate_data(data_path)

    if not success:
        sys.exit(1)
