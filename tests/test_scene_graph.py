"""Tests for the scene graph: given attributes, named relationships, and the object
identity they depend on.

The acceptance case of #78 runs here as written: define *species* for *Coral
fragment*, set it on twenty fragments in one go, and see it on each. Around it: "not
set" stays distinct from "cannot tell" and from false/0, a locked vocabulary refuses
new values, and everything hanging off an object survives an outline edit and comes
back with an undone delete.
"""
import asyncio

import pytest
from iquana_toolbox.schemas.database.contours import Contour
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.database import database, import_models
from app.database.contours import Contours, save_contour_tree
from app.database.datasets import Datasets
from app.database.images import Images
from app.database.labels import Labels
from app.database.masks import Masks
from app.database.scene_graph import ObjectAttributes, ObjectRelationships
from app.database.users import Users
from app.exceptions import InvalidSceneGraphError, SceneGraphNotFoundError
from app.services import hierarchy_cache
from app.services.database_access import annotation_history as history_db
from app.services.database_access import contours as contours_db
from app.services.database_access import scene_graph as graph_db

WIDTH, HEIGHT = 1000, 1000
ANN = "ann"


@event.listens_for(Engine, "connect")
def _fk_pragma(dbapi_connection, connection_record):
    import sqlite3
    if isinstance(dbapi_connection, sqlite3.Connection):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


@pytest.fixture
def db(tmp_path):
    import_models()
    engine = create_engine(f"sqlite:///{tmp_path / 'graph.db'}")
    database.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    hierarchy_cache.clear()
    try:
        yield session
    finally:
        hierarchy_cache.clear()
        session.close()
        engine.dispose()


def _dataset(db, name="reef"):
    ds = Datasets(name=name, description="", dataset_type="image", folder_path=f"/tmp/{name}",
                  created_by=ANN)
    db.add(ds)
    db.flush()
    fragment = Labels(dataset_id=ds.id, name=f"Coral fragment {name}", value=1)
    db.add(fragment)
    db.flush()
    polyp = Labels(dataset_id=ds.id, name=f"Polyp {name}", value=2, parent_id=fragment.id)
    substrate = Labels(dataset_id=ds.id, name=f"Substrate {name}", value=3)
    db.add_all([polyp, substrate])
    image = Images(dataset_id=ds.id, file_name=f"{name}.png", file_path=f"/tmp/{name}.png",
                   thumbnail_file_path="/tmp/t.png", width=WIDTH, height=HEIGHT,
                   color_mode="RGB", scale_x=1.0, scale_y=1.0, unit="px")
    db.add(image)
    db.flush()
    mask = Masks(image_id=image.id, fully_annotated=False, file_path="/tmp/m.png")
    db.add(mask)
    db.commit()
    return {"dataset": ds, "image": image, "mask": mask, "fragment": fragment, "polyp": polyp,
            "substrate": substrate}


@pytest.fixture
def world(db):
    db.add(Users(username=ANN, hashed_password="x", is_admin=False))
    db.commit()
    return _dataset(db)


def _add(db, world, cx=100, cy=100, half=20, label="fragment", parent_id=None):
    x_px = [cx - half, cx + half, cx + half, cx - half]
    y_px = [cy - half, cy - half, cy + half, cy + half]
    contour = Contour(x=[x / WIDTH for x in x_px], y=[y / HEIGHT for y in y_px], added_by="User",
                      confidence=1.0, label_id=world[label].id if label else None,
                      parent_id=parent_id)
    row = save_contour_tree(db, contour, world["mask"].id, parent_id=parent_id, author_username=ANN)
    db.commit()
    return row


def _species(db, world, **kwargs):
    definition = graph_db.create_attribute(
        db, world["dataset"].id, "species", options=["Acropora", "Pocillopora"],
        label_ids=[world["fragment"].id], username=ANN, **kwargs)
    db.commit()
    return definition


def _value(db, contour_id, attribute_id):
    return db.query(ObjectAttributes).filter_by(contour_id=contour_id,
                                                attribute_id=attribute_id).first()


# --------------------------------------------------------------------------- #
# Attributes
# --------------------------------------------------------------------------- #
def test_species_is_set_on_twenty_fragments_in_one_go(db, world):
    species = _species(db, world)
    fragments = [_add(db, world, cx=50 + 45 * i, cy=100) for i in range(20)]

    changes = graph_db.set_attribute_values(db, world["dataset"].id, species.id,
                                            [f.id for f in fragments], "Acropora", username=ANN)
    db.commit()

    assert len(changes) == 20
    values = graph_db.attributes_of_contours(db, [f.id for f in fragments])
    assert all(values[f.id][0]["value"] == "Acropora" for f in fragments)
    assert all(values[f.id][0]["author_username"] == ANN for f in fragments)


def test_not_set_cannot_tell_and_false_are_three_different_things(db, world):
    bleached = graph_db.create_attribute(db, world["dataset"].id, "bleached", value_type="boolean")
    unset, unknown, no = (_add(db, world, cx=100 + 100 * i) for i in range(3))

    graph_db.set_attribute_values(db, world["dataset"].id, bleached.id, [unknown.id], None, username=ANN)
    graph_db.set_attribute_values(db, world["dataset"].id, bleached.id, [no.id], False, username=ANN)
    db.commit()

    assert _value(db, unset.id, bleached.id) is None
    assert _value(db, unknown.id, bleached.id).value is None
    assert _value(db, no.id, bleached.id).value == "false"
    assert _value(db, no.id, bleached.id).value_num == 0.0
    listed = graph_db.attributes_of_contours(db, [unknown.id])[unknown.id][0]
    assert listed["cannot_tell"] is True


def test_a_locked_vocabulary_refuses_a_new_value(db, world):
    species = _species(db, world)
    fragment = _add(db, world)
    with pytest.raises(InvalidSceneGraphError, match="not an allowed value"):
        graph_db.set_attribute_values(db, world["dataset"].id, species.id, [fragment.id],
                                      "Porites", username=ANN)


def test_an_attribute_is_refused_on_objects_outside_its_labels(db, world):
    species = _species(db, world)
    rock = _add(db, world, label="substrate")
    with pytest.raises(InvalidSceneGraphError, match="does not apply"):
        graph_db.set_attribute_values(db, world["dataset"].id, species.id, [rock.id],
                                      "Acropora", username=ANN)


def test_relabelling_out_of_scope_keeps_the_value_but_flags_it(db, world):
    species = _species(db, world)
    fragment = _add(db, world)
    graph_db.set_attribute_values(db, world["dataset"].id, species.id, [fragment.id], "Acropora",
                                  username=ANN)
    fragment.label_id = world["substrate"].id
    db.commit()

    listed = graph_db.attributes_of_contours(db, [fragment.id])[fragment.id][0]
    assert listed["value"] == "Acropora"
    assert listed["out_of_scope"] is True


def test_objects_of_another_dataset_cannot_be_reached(db, world):
    other = _dataset(db, "other")
    species = _species(db, world)
    foreign = _add(db, other)
    with pytest.raises(SceneGraphNotFoundError):
        graph_db.set_attribute_values(db, world["dataset"].id, species.id, [foreign.id],
                                      "Acropora", username=ANN)


def test_retyping_is_refused_when_stored_values_do_not_fit(db, world):
    depth = graph_db.create_attribute(db, world["dataset"].id, "depth class", value_type="text")
    fragment = _add(db, world)
    graph_db.set_attribute_values(db, world["dataset"].id, depth.id, [fragment.id], "shallow",
                                  username=ANN)
    with pytest.raises(InvalidSceneGraphError, match="shallow"):
        graph_db.update_attribute(db, world["dataset"].id, depth.id, value_type="number")


def test_renaming_an_option_rewrites_every_object(db, world):
    species = _species(db, world)
    fragment = _add(db, world)
    graph_db.set_attribute_values(db, world["dataset"].id, species.id, [fragment.id], "Acropora",
                                  username=ANN)
    graph_db.update_attribute(db, world["dataset"].id, species.id,
                              rename_options={"Acropora": "Acropora sp."})
    db.commit()

    assert _value(db, fragment.id, species.id).value == "Acropora sp."
    assert "Acropora sp." in species.option_values


def test_attribute_names_are_unique_per_dataset(db, world):
    _species(db, world)
    with pytest.raises(InvalidSceneGraphError, match="already exists"):
        graph_db.create_attribute(db, world["dataset"].id, " species ")


# --------------------------------------------------------------------------- #
# Relationships
# --------------------------------------------------------------------------- #
def test_selecting_several_objects_fans_out_from_the_subject(db, world):
    a, b, c = (_add(db, world, cx=100 + 100 * i) for i in range(3))
    edges, created = graph_db.add_relationships(db, world["dataset"].id, a.id, [b.id, c.id],
                                                type_name="points_at", username=ANN)
    db.commit()

    assert {(e.subject_id, e.object_id) for e in edges} == {(a.id, b.id), (a.id, c.id)}
    assert len(created) == 2
    assert [t.name for t in graph_db.list_relationship_types(db, world["dataset"].id)] == ["points_at"]


def test_a_typed_name_is_reused_and_existing_edges_are_not_duplicated(db, world):
    a, b = _add(db, world, cx=100), _add(db, world, cx=300)
    graph_db.add_relationships(db, world["dataset"].id, a.id, [b.id], type_name="attached_to")
    _, created = graph_db.add_relationships(db, world["dataset"].id, a.id, [b.id],
                                            type_name="attached_to")
    assert created == []
    assert len(graph_db.list_relationship_types(db, world["dataset"].id)) == 1


def test_symmetric_relationships_are_stored_once(db, world):
    a, b = _add(db, world, cx=100), _add(db, world, cx=300)
    touches = graph_db.create_relationship_type(db, world["dataset"].id, "touches", symmetric=True)
    graph_db.add_relationships(db, world["dataset"].id, b.id, [a.id], type_id=touches.id)
    _, created = graph_db.add_relationships(db, world["dataset"].id, a.id, [b.id], type_id=touches.id)
    assert created == []
    edge = db.query(ObjectRelationships).one()
    assert (edge.subject_id, edge.object_id) == (min(a.id, b.id), max(a.id, b.id))


def test_making_a_type_symmetric_folds_its_mirrored_edges(db, world):
    a, b = _add(db, world, cx=100), _add(db, world, cx=300)
    graph_db.add_relationships(db, world["dataset"].id, a.id, [b.id], type_name="touches")
    graph_db.add_relationships(db, world["dataset"].id, b.id, [a.id], type_name="touches")
    rel_type = graph_db.find_relationship_type(db, world["dataset"].id, "touches")
    graph_db.update_relationship_type(db, world["dataset"].id, rel_type.id, symmetric=True)
    db.commit()
    assert db.query(ObjectRelationships).count() == 1


def test_an_object_cannot_relate_to_itself(db, world):
    a = _add(db, world)
    with pytest.raises(InvalidSceneGraphError):
        graph_db.add_relationships(db, world["dataset"].id, a.id, [a.id], type_name="sees")


def test_deleting_an_object_removes_its_attributes_and_relationships(db, world):
    species = _species(db, world)
    a, b = _add(db, world, cx=100), _add(db, world, cx=300)
    graph_db.set_attribute_values(db, world["dataset"].id, species.id, [a.id], "Acropora")
    graph_db.add_relationships(db, world["dataset"].id, a.id, [b.id], type_name="attached_to")
    db.commit()

    asyncio.run(contours_db.delete_contour(a.id, db))
    assert db.query(ObjectAttributes).count() == 0
    assert db.query(ObjectRelationships).count() == 0


# --------------------------------------------------------------------------- #
# Identity: outline edits keep the graph; undo brings it back
# --------------------------------------------------------------------------- #
def test_an_outline_edit_keeps_children_attributes_and_relationships(db, world):
    species = _species(db, world)
    fragment = _add(db, world, cx=200, cy=200, half=100)
    polyp = _add(db, world, cx=200, cy=200, half=10, label="polyp", parent_id=fragment.id)
    rock = _add(db, world, cx=600, cy=600)
    graph_db.set_attribute_values(db, world["dataset"].id, species.id, [fragment.id], "Acropora")
    graph_db.add_relationships(db, world["dataset"].id, fragment.id, [rock.id],
                               type_name="attached_to")
    db.commit()
    fragment_id, polyp_id = fragment.id, polyp.id

    refined = Contour(x=[0.08, 0.32, 0.32, 0.08], y=[0.08, 0.08, 0.32, 0.32], added_by="sam2",
                      confidence=0.9, label_id=world["fragment"].id)
    assert asyncio.run(contours_db.replace_contour(fragment_id, refined, db))

    db.expire_all()
    assert db.get(Contours, polyp_id).parent_id == fragment_id
    assert _value(db, fragment_id, species.id).value == "Acropora"
    assert db.query(ObjectRelationships).filter_by(subject_id=fragment_id).count() == 1
    assert db.get(Contours, fragment_id).x == [0.08, 0.32, 0.32, 0.08]


def test_undoing_a_delete_brings_back_attributes_and_relationships(db, world):
    species = _species(db, world)
    fragment = _add(db, world, cx=200, cy=200, half=100)
    polyp = _add(db, world, cx=200, cy=200, half=10, label="polyp", parent_id=fragment.id)
    rock = _add(db, world, cx=600, cy=600)
    graph_db.set_attribute_values(db, world["dataset"].id, species.id, [fragment.id], "Pocillopora")
    graph_db.add_relationships(db, world["dataset"].id, rock.id, [polyp.id], type_name="shades")
    db.commit()
    fragment_id, polyp_id, rock_id = fragment.id, polyp.id, rock.id

    snapshot = history_db.snapshot_subtree(fragment_id, db)
    asyncio.run(contours_db.delete_contour(fragment_id, db))
    history_db.record_delete(db, world["mask"].id, ANN, snapshot)
    assert db.query(ObjectRelationships).count() == 0

    history_db.undo(world["mask"].id, ANN, db)

    assert _value(db, fragment_id, species.id).value == "Pocillopora"
    edge = db.query(ObjectRelationships).one()
    assert (edge.subject_id, edge.object_id) == (rock_id, polyp_id)


def test_attribute_changes_are_undoable(db, world):
    species = _species(db, world)
    fragment = _add(db, world)
    for value in ("Acropora", "Pocillopora"):
        changes = graph_db.set_attribute_values(db, world["dataset"].id, species.id,
                                                [fragment.id], value, username=ANN)
        db.commit()
        history_db.record_attribute_changes(db, ANN, changes)

    history_db.undo(world["mask"].id, ANN, db)
    assert _value(db, fragment.id, species.id).value == "Acropora"
    history_db.undo(world["mask"].id, ANN, db)
    assert _value(db, fragment.id, species.id) is None
    history_db.redo(world["mask"].id, ANN, db)
    assert _value(db, fragment.id, species.id).value == "Acropora"


def test_relationship_changes_are_undoable(db, world):
    a, b = _add(db, world, cx=100), _add(db, world, cx=300)
    _, created = graph_db.add_relationships(db, world["dataset"].id, a.id, [b.id],
                                            type_name="points_at", username=ANN)
    db.commit()
    history_db.record_relationship_changes(db, ANN, created=created, deleted=[])

    history_db.undo(world["mask"].id, ANN, db)
    assert db.query(ObjectRelationships).count() == 0
    history_db.redo(world["mask"].id, ANN, db)
    assert db.query(ObjectRelationships).one().id == created[0]["id"]


# --------------------------------------------------------------------------- #
# Split and merge
# --------------------------------------------------------------------------- #
def test_a_merge_moves_relationships_and_fills_missing_attributes(db, world):
    species = _species(db, world)
    survivor, absorbed, rock = (_add(db, world, cx=100 + 200 * i) for i in range(3))
    graph_db.set_attribute_values(db, world["dataset"].id, species.id, [absorbed.id], "Acropora")
    graph_db.add_relationships(db, world["dataset"].id, absorbed.id, [rock.id], type_name="attached_to")
    graph_db.add_relationships(db, world["dataset"].id, survivor.id, [rock.id], type_name="attached_to")
    graph_db.add_relationships(db, world["dataset"].id, survivor.id, [absorbed.id], type_name="touches")

    result = graph_db.transfer(db, world["dataset"].id, [absorbed.id], survivor.id,
                               move_relationships=True)
    db.commit()

    assert result == {"attributes_copied": 1, "relationships_moved": 0, "relationships_dropped": 2}
    assert _value(db, survivor.id, species.id).value == "Acropora"
    assert [(e.subject_id, e.object_id) for e in db.query(ObjectRelationships)] == [(survivor.id, rock.id)]


def test_a_split_copies_attributes_but_leaves_relationships(db, world):
    species = _species(db, world)
    original, half, rock = (_add(db, world, cx=100 + 200 * i) for i in range(3))
    graph_db.set_attribute_values(db, world["dataset"].id, species.id, [original.id], "Acropora")
    graph_db.add_relationships(db, world["dataset"].id, original.id, [rock.id], type_name="attached_to")

    graph_db.transfer(db, world["dataset"].id, [original.id], half.id, move_relationships=False)
    db.commit()

    assert _value(db, half.id, species.id).value == "Acropora"
    assert db.query(ObjectRelationships).one().subject_id == original.id


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def test_the_image_scene_graph_lists_objects_and_both_kinds_of_edge(db, world):
    species = _species(db, world)
    fragment = _add(db, world, cx=200, cy=200, half=100)
    polyp = _add(db, world, cx=200, cy=200, half=10, label="polyp", parent_id=fragment.id)
    rock = _add(db, world, cx=600, cy=600, label="substrate")
    graph_db.set_attribute_values(db, world["dataset"].id, species.id, [fragment.id], "Acropora")
    graph_db.add_relationships(db, world["dataset"].id, fragment.id, [rock.id], type_name="attached_to")
    db.commit()

    graph = graph_db.image_scene_graph(db, world["image"].id)

    assert {o["id"] for o in graph["objects"]} == {fragment.id, polyp.id, rock.id}
    by_id = {o["id"]: o for o in graph["objects"]}
    assert by_id[fragment.id]["attributes"][0]["value"] == "Acropora"
    assert "area" in by_id[fragment.id]["metrics"]
    kinds = {(e["type"], e["subject_id"], e["object_id"]) for e in graph["relationships"]}
    assert kinds == {("part_of", polyp.id, fragment.id), ("attached_to", fragment.id, rock.id)}


# --------------------------------------------------------------------------- #
# Export, grouping and querying
# --------------------------------------------------------------------------- #
def _tag_image(db, world, key, value):
    from app.database.dataset_metadata_keys import DatasetMetadataKeys
    from app.database.image_metadata import ImageMetadata
    if db.query(DatasetMetadataKeys).filter_by(dataset_id=world["dataset"].id, key=key).first() is None:
        db.add(DatasetMetadataKeys(dataset_id=world["dataset"].id, key=key, value_type="categorical",
                                   options=[]))
    db.add(ImageMetadata(image_id=world["image"].id, key=key, value=value))
    db.commit()


@pytest.fixture
def reef(db, world):
    """Three fragments (two Acropora, one cannot-tell), a rock, one attached_to edge."""
    species = _species(db, world)
    a1, a2, unknown = (_add(db, world, cx=100 + 150 * i, half=20 + 10 * i) for i in range(3))
    rock = _add(db, world, cx=800, cy=800, label="substrate")
    graph_db.set_attribute_values(db, world["dataset"].id, species.id, [a1.id, a2.id], "Acropora")
    graph_db.set_attribute_values(db, world["dataset"].id, species.id, [unknown.id], None)
    graph_db.add_relationships(db, world["dataset"].id, a1.id, [rock.id], type_name="attached_to")
    db.commit()
    _tag_image(db, world, "treatment", "heat")
    return {"species": species, "a1": a1, "a2": a2, "unknown": unknown, "rock": rock}


def test_the_export_carries_attribute_and_relationship_columns(db, world, reef):
    from app.services.database_access.datasets import get_dataset_as_df
    df = asyncio.run(get_dataset_as_df(world["dataset"].id, False, False, db))
    rows = df.set_index("contour_id")

    assert rows.loc[reef["a1"].id, "attr_species"] == "Acropora"
    assert rows.loc[reef["unknown"].id, "attr_species"] == graph_db.CANNOT_TELL
    assert rows.loc[reef["a1"].id, "rel_attached_to"] == str(reef["rock"].id)
    assert rows.loc[reef["rock"].id, "rel_attached_to_of"] == str(reef["a1"].id)
    assert rows.loc[reef["rock"].id, "meta_treatment"] == "heat"


def test_the_summary_groups_by_an_attribute(db, world, reef):
    from app.services.database_access.datasets import get_quantification_summary
    summary = asyncio.run(get_quantification_summary(
        world["dataset"].id, False, False, db, group_by_key="attr:species"))

    fragment = str(world["fragment"].id)
    assert summary["groups"]["Acropora"][fragment]["area"]["components"][0]["count"] == 2
    assert summary["groups"][graph_db.CANNOT_TELL][fragment]["area"]["components"][0]["count"] == 1
    # The rock has no species: it lands in the untagged bucket, not in a species.
    assert str(world["substrate"].id) in summary["groups"]["(untagged)"]


def test_only_categorical_and_boolean_attributes_can_group(db, world):
    graph_db.create_attribute(db, world["dataset"].id, "length", value_type="number")
    with pytest.raises(InvalidSceneGraphError, match="too many"):
        graph_db.assert_groupable_attribute(db, world["dataset"].id, "length")


def test_a_query_combines_label_attribute_metadata_and_relationship(db, world, reef):
    from app.services.object_query import Condition, ObjectQuery, RelatedCondition, matching_rows
    query = ObjectQuery(
        label_ids=[world["fragment"].id],
        where=[Condition(attribute="species", value="Acropora"),
               Condition(metadata="treatment", value="heat")],
        related=[RelatedCondition(type="attached_to", label_id=world["substrate"].id)],
    )
    assert [row[0].id for row in matching_rows(db, world["dataset"].id, query)] == [reef["a1"].id]

    query.related[0].negate = True
    assert [row[0].id for row in matching_rows(db, world["dataset"].id, query)] == [reef["a2"].id]


def test_a_query_tells_unset_from_cannot_tell(db, world, reef):
    from app.services.object_query import Condition, ObjectQuery, matching_rows
    fragment_only = [world["fragment"].id]
    cannot_tell = ObjectQuery(label_ids=fragment_only, where=[Condition(attribute="species", op="cannot_tell")])
    not_set = ObjectQuery(where=[Condition(attribute="species", op="not_set")])

    assert [r[0].id for r in matching_rows(db, world["dataset"].id, cannot_tell)] == [reef["unknown"].id]
    assert [r[0].id for r in matching_rows(db, world["dataset"].id, not_set)] == [reef["rock"].id]


def test_a_query_refuses_what_the_dataset_does_not_have(db, world, reef):
    from app.services.object_query import Condition, ObjectQuery, matching_rows
    with pytest.raises(InvalidSceneGraphError, match="no attribute"):
        matching_rows(db, world["dataset"].id, ObjectQuery(where=[Condition(attribute="colour", value="red")]))
    with pytest.raises(InvalidSceneGraphError, match="needs a number"):
        matching_rows(db, world["dataset"].id,
                      ObjectQuery(where=[Condition(attribute="species", op="gt", value=3)]))


def test_running_a_query_returns_the_export_table(db, world, reef):
    from app.services.object_query import Condition, ObjectQuery, run_query
    df = run_query(db, world["dataset"].id,
                   ObjectQuery(where=[Condition(attribute="species", value="Acropora")],
                               metrics=["area"]))
    assert sorted(df["contour_id"]) == sorted([reef["a1"].id, reef["a2"].id])
    assert {"attr_species", "rel_attached_to", "meta_treatment", "area"} <= set(df.columns)
    assert (df["area"] > 0).all()


# --------------------------------------------------------------------------- #
# Filtering on measured attributes (metrics)
# --------------------------------------------------------------------------- #
def _second_image(db, world, scale, unit):
    """Another image in the same dataset, with its own scale."""
    image = Images(dataset_id=world["dataset"].id, file_name="b.png", file_path="/tmp/b.png",
                   thumbnail_file_path="/tmp/tb.png", width=WIDTH, height=HEIGHT, color_mode="RGB",
                   scale_x=scale, scale_y=scale, unit=unit)
    db.add(image)
    db.flush()
    mask = Masks(image_id=image.id, fully_annotated=False, file_path="/tmp/mb.png")
    db.add(mask)
    db.commit()
    return {**world, "image": image, "mask": mask}


def _ids(db, world, **query):
    from app.services.object_query import ObjectQuery, matching_rows
    return [row[0].id for row in matching_rows(db, world["dataset"].id, ObjectQuery(**query))]


def test_an_area_threshold_is_physical_and_uses_each_images_own_scale(db, world):
    from app.services.object_query import Condition
    world["image"].scale_x = world["image"].scale_y = 0.1   # 0.1 mm per pixel
    world["image"].unit = "mm"
    db.commit()
    other = _second_image(db, world, scale=0.05, unit="mm")
    # The same 40 x 40 px square on both: 16 mm² on the first image, 4 mm² on the second.
    coarse = _add(db, world, cx=100, half=20)
    fine = _add(db, other, cx=100, half=20)

    assert _ids(db, world, where=[Condition(metric="area", op="gt", value=10)]) == [coarse.id]
    assert _ids(db, world, where=[Condition(metric="area", op="lte", value=10)]) == [fine.id]
    # A length scales by one factor: 40 px wide = 4 mm and 2 mm.
    assert _ids(db, world, where=[Condition(metric="perimeter", op="lt", value=10)]) == [fine.id]


def test_with_mixed_units_thresholds_are_in_pixels(db, world):
    from app.services.object_query import Condition, ObjectQuery, scale_status
    other = _second_image(db, world, scale=0.05, unit="mm")   # world's image stays in px
    small = _add(db, world, cx=100, half=10)    # 400 px²
    large = _add(db, other, cx=100, half=30)    # 3600 px²

    assert scale_status(db, world["dataset"].id, ObjectQuery())["display_unit"] == "px"
    assert _ids(db, world, where=[Condition(metric="area", op="gt", value=1000)]) == [large.id]
    assert small.id not in _ids(db, world, where=[Condition(metric="area", op="gt", value=1000)])


def test_a_colour_component_can_be_filtered(db, world):
    from app.database.contour_metrics import ContourMetrics
    from app.services.object_query import Condition
    pale, dark = _add(db, world, cx=100), _add(db, world, cx=300)
    for contour, lab in ((pale, (80.0, 2.0, 10.0)), (dark, (30.0, 5.0, 20.0))):
        for component, value in enumerate(lab):
            db.add(ContourMetrics(contour_id=contour.id, metric_key="mean_color_lab", component=component,
                                  value=value, unit="", stale=False))
    db.commit()

    assert _ids(db, world, where=[Condition(metric="mean_color_lab", component=0, op="lt", value=40)]) == [dark.id]
    assert _ids(db, world, where=[Condition(metric="mean_color_lab", component=2, op="lt", value=15)]) == [pale.id]


def test_stale_and_missing_metric_values_do_not_match(db, world):
    from app.database.contour_metrics import ContourMetrics
    from app.services.object_query import Condition
    measured, stale, unmeasured = (_add(db, world, cx=100 + 200 * i) for i in range(3))
    db.add(ContourMetrics(contour_id=measured.id, metric_key="mean_intensity", component=0,
                          value=0.5, unit="", stale=False))
    db.add(ContourMetrics(contour_id=stale.id, metric_key="mean_intensity", component=0,
                          value=0.5, unit="", stale=True))
    db.commit()

    assert _ids(db, world, where=[Condition(metric="mean_intensity", op="gte", value=0)]) == [measured.id]
    assert _ids(db, world, where=[Condition(metric="mean_intensity", op="not_set")]) == [stale.id, unmeasured.id]


def test_metric_conditions_are_checked(db, world):
    from pydantic import ValidationError
    from app.services.object_query import Condition
    with pytest.raises(ValidationError, match="metric condition"):
        Condition(metric="area", op="in", values=[1, 2])
    with pytest.raises(ValidationError, match="exactly one"):
        Condition(metric="area", attribute="species", op="gt", value=1)
    with pytest.raises(InvalidSceneGraphError, match="Unknown metric"):
        _ids(db, world, where=[Condition(metric="volume", op="gt", value=1)])
    with pytest.raises(InvalidSceneGraphError, match="component 3 does not exist"):
        _ids(db, world, where=[Condition(metric="mean_color_lab", component=3, op="gt", value=1)])
    with pytest.raises(InvalidSceneGraphError, match="not a number"):
        _ids(db, world, where=[Condition(metric="area", op="gt", value="big")])


def test_running_a_metric_query_reports_its_unit(db, world):
    from app.services.object_query import Condition, ObjectQuery, run_query
    big, small = _add(db, world, cx=100, half=40), _add(db, world, cx=400, half=10)
    df = run_query(db, world["dataset"].id,
                   ObjectQuery(where=[Condition(metric="area", op="gt", value=1000)], metrics=["area"]))
    assert list(df["contour_id"]) == [big.id]
    assert df.attrs["scale_status"]["display_unit"] == "px"


# --------------------------------------------------------------------------- #
# A model class can stand for a label *and* an attribute value
# --------------------------------------------------------------------------- #
SPECIES_MODEL = "coral_species_detector"
SPECIES_INFO = {
    "registry_key": SPECIES_MODEL,
    "name": "Coral species detector",
    "classes": [{"key": "acropora", "name": "Acropora"}, {"key": "pocillopora", "name": "Pocillopora"}],
}


def test_a_species_class_maps_to_the_fragment_label_and_a_suggested_species(db, world, monkeypatch):
    from app.schemas.model_label_mapping import ClassMappingWrite
    from app.services import model_label_mapping as mlm
    monkeypatch.setattr(mlm, "_model_info", lambda key: SPECIES_INFO)
    species = _species(db, world)
    mlm.save_mapping(db, world["dataset"].id, SPECIES_MODEL, [
        ClassMappingWrite(model_class="acropora", action="map", label_id=world["fragment"].id,
                          attribute_id=species.id),
        ClassMappingWrite(model_class="pocillopora", action="map", label_id=world["fragment"].id,
                          attribute_id=species.id),
    ], ANN)

    predicted = [Contour(x=[0.1, 0.2, 0.2, 0.1], y=[0.1, 0.1, 0.2, 0.2], confidence=0.8,
                         model_class="acropora")]
    labelled = mlm.apply_mapping(db, world["dataset"].id, SPECIES_MODEL, predicted)
    assert labelled[0].label_id == world["fragment"].id
    saved = save_contour_tree(db, labelled[0], world["mask"].id, author_username=ANN)
    assert mlm.apply_class_attributes(db, world["dataset"].id, SPECIES_MODEL, [saved.id]) == 1
    db.commit()

    value = _value(db, saved.id, species.id)
    assert (value.value, value.status, value.source, value.model_key) == (
        "Acropora", "suggested", "model", SPECIES_MODEL)
    # Suggestions are not exported until someone confirms them.
    assert graph_db.attribute_columns(db, world["dataset"].id, [saved.id])[0] == {}
    graph_db.confirm_attribute_values(db, world["dataset"].id, species.id, [saved.id], ANN)
    assert graph_db.attribute_columns(db, world["dataset"].id, [saved.id])[0][saved.id] == {
        "species": "Acropora"}


def test_a_class_value_outside_the_vocabulary_is_refused(db, world, monkeypatch):
    from fastapi import HTTPException
    from app.schemas.model_label_mapping import ClassMappingWrite
    from app.services import model_label_mapping as mlm
    monkeypatch.setattr(mlm, "_model_info", lambda key: SPECIES_INFO)
    species = _species(db, world)
    with pytest.raises(HTTPException, match="not an allowed value"):
        mlm.save_mapping(db, world["dataset"].id, SPECIES_MODEL, [
            ClassMappingWrite(model_class="acropora", action="map", label_id=world["fragment"].id,
                              attribute_id=species.id, attribute_value="Porites"),
        ], ANN)


# --------------------------------------------------------------------------- #
# Folding is-a labels into an attribute
# --------------------------------------------------------------------------- #
def test_species_labels_fold_into_a_fragment_label_and_a_species_attribute(db, world):
    from app.database.quantification_profiles import QuantificationProfiles
    from app.services.label_to_attribute import convert_labels_to_attribute
    ds = world["dataset"].id
    acropora = Labels(dataset_id=ds, name="Acropora", value=10)
    pocillopora = Labels(dataset_id=ds, name="Pocillopora", value=11)
    db.add_all([acropora, pocillopora])
    db.flush()
    tentacle = Labels(dataset_id=ds, name="Tentacle", value=12, parent_id=acropora.id)
    db.add(tentacle)
    db.add(QuantificationProfiles(dataset_id=ds, name="corals", entries=[
        {"metric_key": "area", "params": {}, "label_ids": [acropora.id, pocillopora.id]}]))
    db.commit()
    a = _add(db, world, cx=100, label=None)
    p = _add(db, world, cx=300, label=None)
    a.label_id, p.label_id = acropora.id, pocillopora.id
    db.commit()
    child = _add(db, world, cx=100, half=5, label=None, parent_id=a.id)
    child.label_id = tentacle.id
    db.commit()
    a_id, p_id, child_id, acropora_id, pocillopora_id = a.id, p.id, child.id, acropora.id, pocillopora.id

    preview = convert_labels_to_attribute(db, ds, [acropora_id, pocillopora_id], "species",
                                          target_label_name="Coral", dry_run=True)
    assert preview.objects_relabelled == 2 and preview.child_labels_moved == ["Tentacle"]
    assert db.query(Labels).filter_by(name="Coral").first() is None

    report = convert_labels_to_attribute(db, ds, [acropora_id, pocillopora_id], "species",
                                         target_label_name="Coral", dry_run=False)
    db.commit()
    db.expire_all()

    coral = db.query(Labels).filter_by(dataset_id=ds, name="Coral").one()
    species = graph_db.find_attribute(db, ds, "species")
    assert species.option_values == ["Acropora", "Pocillopora"]
    assert [label.id for label in species.labels] == [coral.id]
    assert db.get(Contours, a_id).label_id == coral.id
    assert _value(db, a_id, species.id).value == "Acropora"
    assert _value(db, p_id, species.id).value == "Pocillopora"
    assert db.get(Contours, child_id).parent_id == a_id
    assert db.query(Labels).filter_by(name="Tentacle").one().parent_id == coral.id
    assert db.query(Labels).filter(Labels.id.in_([acropora_id, pocillopora_id])).count() == 0
    profile = db.query(QuantificationProfiles).filter_by(name="corals").one()
    assert profile.entries[0]["label_ids"] == [coral.id]
    assert report.values_written == 2 and report.conflicts == []


def test_folding_refuses_labels_with_different_parents(db, world):
    from app.services.label_to_attribute import convert_labels_to_attribute
    with pytest.raises(InvalidSceneGraphError, match="share a parent"):
        convert_labels_to_attribute(db, world["dataset"].id, [world["fragment"].id, world["polyp"].id],
                                    "kind", target_label_name="Thing", dry_run=True)


def test_an_approved_label_space_draft_creates_its_attributes(db, world):
    from app.schemas.label_space import DraftAttribute, DraftLabel
    from app.services.database_access.labels import bulk_create_labels
    draft = [
        DraftLabel(name="Coral Fragment", attributes=[
            DraftAttribute(name="species", options=["Acropora", "Pocillopora"]),
            DraftAttribute(name="health", options=["healthy", "bleached"]),
        ], children=[DraftLabel(name="Coral Polyp", attributes=[
            DraftAttribute(name="health", options=["healthy", "dead"])])]),
    ]
    assert asyncio.run(bulk_create_labels(world["dataset"].id, draft, db)) == 2

    fragment = db.query(Labels).filter_by(dataset_id=world["dataset"].id, name="Coral Fragment").one()
    polyp = db.query(Labels).filter_by(dataset_id=world["dataset"].id, name="Coral Polyp").one()
    species = graph_db.find_attribute(db, world["dataset"].id, "species")
    health = graph_db.find_attribute(db, world["dataset"].id, "health")
    assert [label.id for label in species.labels] == [fragment.id]
    assert sorted(label.id for label in health.labels) == sorted([fragment.id, polyp.id])
    assert health.option_values == ["healthy", "bleached", "dead"]
