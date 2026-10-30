"""Create (or promote) a platform admin from the command line.

There is no self-registration: accounts are created by an admin on the admin page
(``POST /admin/users``), which needs an admin to call it. This is how the *first*
admin gets onto an instance. It talks to the database directly, on purpose: it is
the bootstrap step that runs *before* there is anyone to authenticate as.

Usage::

    python scripts/create_admin.py <username> <password>
    python scripts/create_admin.py --bootstrap <username>

Idempotent: an existing account is promoted to admin and reactivated, and its
password is reset only when one is given.

``--bootstrap`` is what the installer runs. It does nothing when the instance already
has an active admin; otherwise it creates (or promotes) the account with a generated
password, flagged ``must_change_password``, and prints *only that password* on stdout
so the installer can show it once. Generating it here keeps it off the command line,
where every other user of the machine could read it from the process list.

Because it bypasses the API it also bypasses the API's password rules. That is
the right trade for a recovery tool — an operator locked out of their own
instance should not also be locked out by a length check — but it means the
password you pass is the password you get. Choose accordingly.
"""
import argparse
import secrets
import sys

from app.database import get_context_session, init_db
from app.database.users import Users
from app.schemas.permissions import GlobalRole
from app.services.auth import get_password_hash
from app.services.database_access.organizations import join_default_organization


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or promote a platform admin.")
    parser.add_argument("username")
    parser.add_argument("password", nargs="?",
                        help="Omit to promote an existing account without touching its password.")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Installer mode: only if there is no active admin yet; generates the "
                             "password and prints it alone on stdout.")
    args = parser.parse_args()
    if args.bootstrap and args.password:
        parser.error("--bootstrap generates the password; do not pass one.")

    # A fresh deployment may not have been booted through create_app yet.
    init_db()

    if args.bootstrap:
        return bootstrap(args.username)

    with get_context_session() as db:
        account = db.query(Users).filter_by(username=args.username).first()

        if account is None:
            if not args.password:
                print(f"No account named {args.username!r}; pass a password to create one.")
                return 1
            db.add(Users(
                username=args.username,
                hashed_password=get_password_hash(args.password),
                global_role=GlobalRole.ADMIN.value,
                is_active=True,
            ))
            join_default_organization(args.username, db)
            db.commit()
            print(f"Created {args.username!r} as a platform admin.")
            return 0

        was = account.global_role
        account.global_role = GlobalRole.ADMIN.value
        # A deactivated admin is no use as a way back in.
        account.is_active = True
        if args.password:
            account.hashed_password = get_password_hash(args.password)
        db.commit()

        changed = f"promoted from {was}" if was != GlobalRole.ADMIN.value else "already an admin"
        reset = ", password reset" if args.password else ""
        print(f"{args.username!r}: {changed}{reset}.")
        return 0


def bootstrap(username: str) -> int:
    """Create the first admin unless the instance already has one (see module docstring)."""
    with get_context_session() as db:
        has_admin = db.query(Users).filter(
            Users.global_role == GlobalRole.ADMIN.value, Users.is_active.is_(True)
        ).first() is not None
        if has_admin:
            print("An active admin already exists; nothing to do.", file=sys.stderr)
            return 0

        password = secrets.token_urlsafe(12)
        account = db.query(Users).filter_by(username=username).first()
        if account is None:
            account = Users(username=username)
            db.add(account)
            join_default_organization(username, db)
        account.hashed_password = get_password_hash(password)
        account.global_role = GlobalRole.ADMIN.value
        account.is_active = True
        account.must_change_password = True
        db.commit()
    print(f"Created {username!r} as the first platform admin.", file=sys.stderr)
    print(password)
    return 0


if __name__ == "__main__":
    sys.exit(main())
