import pytest

from aloy_backend.models import SkillDefinition, SkillGrant
from aloy_backend.skills import load_skill_catalog
from pori import ToolRegistry

pytestmark = pytest.mark.asyncio


async def test_skill_lifecycle_requires_approval_and_grant(client):
    org = await client.post(
        "/v1/organizations",
        headers={"X-Test-User": "alice"},
        json={"name": "Skills", "slug": "skills-release-d"},
    )
    organization_id = org.json()["id"]
    headers = {"X-Test-User": "alice", "X-Pori-Organization": organization_id}

    created = await client.post(
        "/v1/skills",
        headers=headers,
        json={
            "slug": "database-migration",
            "name": "Database Migration",
            "summary": "Plan migration, verification, and rollback",
            "instructions": "Always include rollback and verification steps.",
            "tags": ["database", "migration"],
            "category": "software-development",
            "author": "Data Team",
            "license": "MIT",
            "commands": ["migrate"],
            "argument_hint": "Which database is being migrated?",
            "provenance": "registry",
            "trust_level": "organization",
            "required_commands": ["psql"],
            "setup_help": "Install PostgreSQL CLI tools.",
            "source_url": "https://example.com/database-migration",
            "install_command": "pori skills install database-migration",
            "readiness_warnings": ["reviewed by policy"],
        },
    )
    assert created.status_code == 201
    skill = created.json()
    assert skill["status"] == "draft"
    assert skill["category"] == "software-development"
    assert skill["commands"] == ["migrate"]
    assert skill["argument_hint"] == "Which database is being migrated?"
    assert skill["provenance"] == "registry"
    assert skill["required_commands"] == ["psql"]
    assert skill["readiness_warnings"] == ["reviewed by policy"]

    approved = await client.patch(
        f"/v1/skills/{skill['id']}",
        headers=headers,
        json={"status": "approved"},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"

    grant = await client.post(
        f"/v1/skills/{skill['id']}/grants",
        headers=headers,
        json={"principal_type": "role", "principal_id": "member"},
    )
    assert grant.status_code == 201

    grants = await client.get(f"/v1/skills/{skill['id']}/grants", headers=headers)
    assert [item["principal_id"] for item in grants.json()] == ["member"]


async def test_cloud_skill_catalog_loads_only_approved_granted_skills(db_session_maker):
    async with db_session_maker() as session:
        approved = SkillDefinition(
            organization_id="org-1",
            created_by="alice",
            slug="database-migration",
            version="1",
            name="Database Migration",
            summary="Plan database migration and rollback",
            instructions="Include verification and rollback.",
            tags=["database"],
            category="software-development",
            author="Data Team",
            license="MIT",
            commands=["migrate"],
            argument_hint="Which database is being migrated?",
            provenance="registry",
            trust_level="organization",
            required_commands=["psql"],
            setup_help="Install PostgreSQL CLI tools.",
            source_url="https://example.com/database-migration",
            install_command="pori skills install database-migration",
            readiness_warnings=["reviewed by policy"],
            status="approved",
        )
        draft = SkillDefinition(
            organization_id="org-1",
            created_by="alice",
            slug="draft-only",
            version="1",
            name="Draft",
            summary="Should not load",
            instructions="Hidden draft text",
            status="draft",
        )
        session.add_all([approved, draft])
        await session.commit()
        await session.refresh(approved)
        session.add(
            SkillGrant(
                organization_id="org-1",
                skill_id=approved.id,
                principal_type="role",
                principal_id="member",
                created_by="alice",
            )
        )
        await session.commit()

        catalog = await load_skill_catalog(
            session,
            organization_id="org-1",
            user_id="bob",
            role="member",
        )

    registry = ToolRegistry()
    selected = catalog.select("database migration", registry.snapshot())
    skills = catalog.load_selected(selected)
    assert [skill.manifest.slug for skill in skills] == ["database-migration"]
    assert skills[0].manifest.category == "software-development"
    assert skills[0].manifest.author == "Data Team"
    assert skills[0].manifest.license == "MIT"
    assert skills[0].manifest.commands == ("migrate",)
    assert skills[0].manifest.argument_hint == "Which database is being migrated?"
    assert skills[0].manifest.provenance == "registry"
    assert skills[0].manifest.trust_level == "organization"
    assert skills[0].manifest.required_commands == ("psql",)
    assert skills[0].manifest.setup_help == "Install PostgreSQL CLI tools."
    assert skills[0].manifest.source_url == "https://example.com/database-migration"
    assert (
        skills[0].manifest.install_command == "pori skills install database-migration"
    )
    assert skills[0].manifest.readiness_warnings == ("reviewed by policy",)
    assert "Hidden draft" not in skills[0].instructions
