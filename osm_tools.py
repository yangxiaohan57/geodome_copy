"""
OSM utility functions
many of the functions are adapted from the osmnx package https://github.com/gboeing/osmnx
"""
# TODO: replace print with logging
import json
import requests
import geopandas as gpd

from shapely.geometry import mapping, Point, Polygon, MultiPolygon, LineString

import math

# set the default coordinate reference system
DEFAULT_CRS = "epsg:4326"

DEFAULT_OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"

DEFAULT_QUERY_TEMPLATE = "[timeout:900][out:json];(node{};<;>;);out;"


def _osm_download(bbox=None, url=DEFAULT_OVERPASS_URL, custom_query=None):
    if bbox is not None:
        # TODO: assert valid bbox
        query = DEFAULT_QUERY_TEMPLATE.format(bbox)
        pass
    else:
        query = custom_query
        pass
    if query is None:
        raise ValueError('you must either pass a valid bounding box or a custome query to the function.')

    response = requests.get(url, params={'data': query})
    return response.json()


def _parse_coords(response):
    """from osmnx.pois"""
    coords = {}
    for result in response["elements"]:
        if "type" in result and result["type"] == "node":
            coords[result["id"]] = {"lat": result["lat"], "lon": result["lon"]}
    return coords


def _parse_osm_node(response):
    """from osmnx.pois"""
    try:
        point = Point(response["lon"], response["lat"])

        poi = {"osmid": response["id"], "geometry": point}

        if "tags" in response:
            for tag in response["tags"]:
                poi[tag] = response["tags"][tag]

    except Exception:
        print(f'Point has invalid geometry: {response["id"]}')

    return poi


def _is_closed_polygon(coords, nodes):
    """
    if the coordinates of the first and last node are the same then
    it is a closed polygon, otherwise it's a line
    """
    first_node_coords = coords[nodes[0]]["lon"], coords[nodes[0]]["lat"]
    last_node_coords = coords[nodes[-1]]["lon"], coords[nodes[-1]]["lat"]
    return (first_node_coords == last_node_coords)


def _parse_polygonal_poi(coords, response, verbose):
    """from osmnx.pois"""
    if "type" in response and response["type"] == "way":
        nodes = response["nodes"]

        try:
            if(_is_closed_polygon(coords, nodes)):
                geometry = Polygon([(coords[node]["lon"], coords[node]["lat"]) for node in nodes])
                pass
            else:
                geometry = LineString([(coords[node]["lon"], coords[node]["lat"]) for node in nodes])
                pass

            poi = {"osmid": response["id"], "nodes": nodes, "geometry": geometry}

            if "tags" in response:
                for tag in response["tags"]:
                    poi[tag] = response["tags"][tag]
            return poi

        except KeyError:
            if(verbose):
                print("Node in Way not included in response")
                pass
            pass

        except Exception as e:
            print(e, f"Polygon has invalid geometry: {nodes}")
            pass


def _parse_osm_relations(relations, df_osm_ways, verbose):
    """from osmnx.pois"""
    gdf_relations = gpd.GeoDataFrame()

    # Iterate over relations and extract the items
    for relation in relations:
        try:
            if relation["tags"]["type"] == "multipolygon":
                # Parse member 'way' ids
                member_way_ids = [member["ref"] for member in relation["members"] if member["type"] == "way"]
                # Extract the ways
                member_ways = df_osm_ways.reindex(member_way_ids)
                # Extract the nodes of those ways
                member_nodes = list(member_ways["nodes"].values)
                try:
                    # Create MultiPolygon from geometries (exclude NaNs)
                    multipoly = MultiPolygon(list(member_ways["geometry"]))
                except Exception:
                    multipoly = _invalid_multipoly_handler(
                        gdf=member_ways, relation=relation, way_ids=member_way_ids
                    )

                if multipoly:
                    # Create GeoDataFrame with the tags and the MultiPolygon and its
                    # 'ways' (ids), and the 'nodes' of those ways
                    geo = gpd.GeoDataFrame(
                        relation["tags"], index=[relation["id"]])
                    # Initialize columns (needed for .loc inserts)
                    geo = geo.assign(
                        geometry=None, ways=None, nodes=None, element_type=None, osmid=None
                    )
                    # Add attributes
                    geo.loc[relation["id"], "geometry"] = multipoly
                    geo.loc[relation["id"], "ways"] = member_way_ids
                    geo.loc[relation["id"], "nodes"] = member_nodes
                    geo.loc[relation["id"], "element_type"] = "relation"
                    geo.loc[relation["id"], "osmid"] = relation["id"]

                    # Append to relation GeoDataFrame
                    gdf_relations = gdf_relations.append(geo, sort=False)
                    # Remove such 'ways' from 'df_osm_ways' that are part of the 'relation'
                    df_osm_ways = df_osm_ways.drop(member_way_ids)
        except Exception:
            if(verbose):
                print(f'Could not parse OSM relation {relation["id"]}')
                pass

    # Merge df_osm_ways and the gdf_relations
    df_osm_ways = df_osm_ways.append(gdf_relations, sort=False)
    return df_osm_ways

def _invalid_multipoly_handler(gdf, relation, way_ids):  # pragma: no cover
    """from osmnx.pois"""
    try:
        gdf_clean = gdf.dropna(subset=["geometry"])
        multipoly = MultiPolygon(list(gdf_clean["geometry"]))
        return multipoly

    except Exception:
        # TODO: redirect error message
        # print(f'Invalid geometry at relation "{relation["id"]}", way IDs: {way_ids}') 
        return None


def _create_gdf(response, crs, verbose):

    # Parse coordinates from all the nodes in the response
    coords = _parse_coords(response)

    # POI nodes
    poi_nodes = {}

    # POI ways
    poi_ways = {}

    # A list of POI relations
    relations = []

    for result in response["elements"]:
        if result["type"] == "node" and "tags" in result:
            poi = _parse_osm_node(response=result)
            # Add element_type
            poi["element_type"] = "node"
            # Add to 'pois'
            poi_nodes[result["id"]] = poi
            pass

        elif result["type"] == "way":
            # Parse POI area Polygon
            poi_area = _parse_polygonal_poi(coords=coords, response=result, verbose=verbose)
            if poi_area:
                # Add element_type
                poi_area["element_type"] = "way"
                # Add to 'poi_ways'
                poi_ways[result["id"]] = poi_area
                pass
            pass

        elif result["type"] == "relation":
            # Add relation to a relation list (needs to be parsed after
            # all nodes and ways have been parsed)
            relations.append(result)

    # Create GeoDataFrames
    gdf_nodes = gpd.GeoDataFrame(poi_nodes).T
    gdf_nodes.crs = crs

    gdf_ways = gpd.GeoDataFrame(poi_ways).T
    gdf_ways.crs = crs

    # Parse relations (MultiPolygons) from 'ways'
    gdf_ways = _parse_osm_relations(relations=relations, df_osm_ways=gdf_ways, verbose=verbose)

    # Combine GeoDataFrames
    gdf = gdf_nodes.append(gdf_ways, sort=False)

    return gdf


def _bbox_from_point(point, dist=1000):
    """
    takes as input a coordinate point (lat, long)
    returns a bounding box (south, west, north, east) centered on that poing with side length dist
    formula from http://www.movable-type.co.uk/scripts/latlong.html#rhumblines
    """
    earth_radius = 6371000  # meters
    angular_distance = math.degrees(0.5 * (dist / earth_radius))

    lat, lon = point
    delta_lat = angular_distance
    delta_lon = angular_distance/math.cos(math.radians(lat))

    south, north = lat - delta_lat, lat + delta_lat
    west , east = lon - delta_lon, lon + delta_lon
    return south, west, north, east


def gdf_from_bbox(bbox, verbose=False):
    """
    takes as input a bounding box, (south, west, north, east)
    returns a GeoDataFrame for data returned from overpass API within that 
    bounding box 
    """
    data = _osm_download(bbox)
    gdf = _create_gdf(data, DEFAULT_CRS, verbose)
    return gdf


def gdf_from_point(point, dist=1000, verbose=False):
    """
    takes as input a coordinate point (lat, long)
    returns a GeoDataFrame for data returned from overpass API within 
    a bounding box centered around that point with side-lenght dist
    """
    bbox = _bbox_from_point(point, dist)
    data = _osm_download(bbox)
    gdf = _create_gdf(data, DEFAULT_CRS, verbose)
    return gdf


def count_tags_in_area(point, dist=1000):
    """
    takes as input a coordinate `point` (lat, long)
    returns a dictionary of available OSM tags and the number of 
    nodes with that tag within a bounding box centered around 
    that point with side-lenght `dist`
    """
    bbox = _bbox_from_point(point, dist)
    data = _osm_download(bbox)

    tag_counts = {}
    for element in data['elements']:
        if('tags' in element):
            for tag_key in element['tags']:
                tag_value = f"{tag_key}>>{element['tags'][tag_key]}"
                tag_counts[tag_key] = tag_counts.get(tag_key, 0)+1
                tag_counts[tag_value] = tag_counts.get(tag_value, 0)+1
                pass
            pass
        pass
    # sort by count
    tag_counts = {tag: count for tag,count in sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)}
    return tag_counts
