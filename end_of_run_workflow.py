import os
import traceback

from prefect import flow, get_run_logger, task
from prefect.task_runners import ConcurrentTaskRunner
from prefect.blocks.notifications import SlackWebhook
from prefect.context import FlowRunContext
from prefect.settings import PREFECT_UI_URL

#from analysis import run_analysis
from data_validation import data_validation_task, get_run
from export import export
from linker import create_symlinks
from dotenv import load_dotenv

CATALOG_NAME = "cms"


def slack(func):
    """
    Send a message to mon-prefect and mon-prefect-cs slack channels if the flow-run failed.
    Send a message to mon-prefect-cms slack channel with the flow-run status.
    Send a message to mon-bluesky slack channel if the bluesky-run failed.

    NOTE: the name of this inner function is the same as the real end_of_workflow() function because
    when the decorator is used, Prefect sees the name of this inner function as the name of
    the flow. To keep the naming of workflows consistent, the name of this inner function had to match the expected name.
    """

    def end_of_run_workflow(stop_doc, api_key=None, dry_run=False):
        flow_run_name = FlowRunContext.get().flow_run.dict().get("name")

        # Load slack credentials that are saved in Prefect.
        mon_prefect = SlackWebhook.load("mon-prefect")
        mon_bluesky = SlackWebhook.load("mon-bluesky")
        mon_prefect_cms = SlackWebhook.load("mon-prefect-cms")
        mon_prefect_cs = SlackWebhook.load("mon-prefect-cs")

        # Get the uid.
        uid = stop_doc["run_start"]

        # Get the scan_id.
        run = get_run(uid, api_key=api_key)
        scan_id = run.start["scan_id"]

        # Send a message to mon-bluesky if bluesky-run failed.
        if stop_doc.get("exit_status") == "fail":
            mon_bluesky.notify(
                f":bangbang: {CATALOG_NAME} bluesky-run failed. (*{flow_run_name}*)\n ```run_start: {uid}\nscan_id: {scan_id}``` ```reason: {stop_doc.get('reason', 'none')}```"
            )

        try:
            result = func(stop_doc, api_key=api_key, dry_run=dry_run)

            # Send a message to mon-prefect-cms if flow-run is successful.
            message = f":white_check_mark: (This is from a test, ignore that if it fails){CATALOG_NAME} flow-run successful. (*{flow_run_name}*)\n ```run_start: {uid}\nscan_id: {scan_id}```"
            mon_prefect_cms.notify(message)
            return result
        except Exception as e:
            tb = traceback.format_exception_only(e)

            # Send a message to mon-prefect-cms, mon-prefect if flow-run failed.
            message = f":bangbang: {CATALOG_NAME} flow-run failed. (*{flow_run_name}*)\n ```run_start: {uid}\nscan_id: {scan_id}``` ```{tb[-1]}```"
            mon_prefect.notify(message)
            mon_prefect_cms.notify(message)
            flow_run = FlowRunContext.get().flow_run
            # Add link to flow-run for the message to mon-prefect-cs.
            program_message = (
                f":bangbang: {CATALOG_NAME} flow-run failed. <{PREFECT_UI_URL.value()}/flow-runs/"
                + f"flow-run/{flow_run.id}|the flow run link> (*{flow_run_name}*)\n ```run_start: {uid}\nscan_id: {scan_id}``` ```{tb[-1]}```"
            )
            mon_prefect_cs.notify(program_message)
            raise

    return end_of_run_workflow


@task
def log_completion():
    logger = get_run_logger()
    logger.info("Complete")


@flow(task_runner=ConcurrentTaskRunner())
@slack
def end_of_run_workflow(stop_doc, api_key=None, dry_run=False):
    load_dotenv()
    logger = get_run_logger()
    uid = stop_doc["run_start"]

    # Launch validation, analysis, and linker tasks concurrently
    linker_task = create_symlinks.submit(uid, api_key=api_key, dry_run=dry_run)
    logger.info("Launched linker task")

    validation_task = data_validation_task.submit(uid, api_key=api_key)
    logger.info("Launched validation tasks")

    export(ref=uid, api_key=api_key, subdirs=True, dry_run=dry_run)
    logger.info("Completed export flow")

    # analysis_task = run_analysis(raw_ref=uid)
    # logger.info("Launched analysis task")

    # Wait for all tasks to comple
    logger.info("Waiting for tasks to complete")
    linker_task.result()
    validation_task.result()
    # analysis_task.result()
    log_completion()
