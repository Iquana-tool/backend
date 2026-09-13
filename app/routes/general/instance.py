from fastapi import APIRouter

from app.services.instance import get_instance_config

router = APIRouter(prefix="/instance", tags=["instance"])


@router.get("/")
def read_instance():
    """Describe this deployment to its sign-in page.

    Deliberately unauthenticated: the whole point is to greet somebody who does
    not have an account yet and tell them whose instance they reached and who to
    ask for access. Nothing here is a secret — it is the same information a
    hosting group would print on the page anyway.
    """
    config = get_instance_config()
    return {
        "success": True,
        "message": "Successfully retrieved instance configuration",
        "result": {
            "name": config.name,
            "organisation": config.organisation,
            "contact": config.contact,
            "notice": config.notice,
            "allow_registration": config.allow_registration,
        },
    }
