"""Create (or promote) a platform admin from the command line.

There is otherwise no way to get the *first* admin onto an instance. Nothing
seeds one; ``POST /auth/register`` only ever makes a plain member and shuts as
soon as any account exists; and ``POST /admin/users`` — the endpoint behind the
admin page — needs an admin to call it. An instance whose accounts were all
created by self-registration therefore has nobody who can administer it, and no
way to fix that from inside the application.

This is the way in. It talks to the database directly, on purpose: it is the
bootstrap step that runs *before* there is anyone to authenticate as.

Usage::

    python scripts/create_admin.py <username> <password>

Idempotent: an existing account is promoted to admin and reactivated, and its
password is reset only when one is given.

Because it bypasses the API it also bypasses the API's password rules. That is
the right trade for a recovery tool — an operator locked out of their own
instance should not also be locked out by a length check — but it means the
password you pass is the password you get. Choose accordingly.
"""
import argparse
import sys

from app.database import get_context_session, init_db
from app.database.users import Users
from app.schemas.permissions import GlobalRole
from app.services.auth import get_password_hash


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or promote a platform admin.")
    parser.add_argument("username")
    parser.add_argument("password", nargs="?",
                        help="Omit to promote an existing account without touching its password.")
    args = parser.parse_args()

    # A fresh deployment may not have been booted through create_app yet.
    init_db()

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


if __name__ == "__main__":
    sys.exit(main())
