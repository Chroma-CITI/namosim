import Box2D as b2
import json


class B2WorldEncoder(json.JSONEncoder):
    """
    Encodes the Box2D world object into a RUBE-compatible json file.
    """
    def default(self, obj):

        if isinstance(obj, b2.b2World):
            world_json = {
                "gravity": self.b2_vec2_to_rube_vec(obj.gravity),
                "allowSleep": obj.allowSleeping,
                "autoClearForces": obj.autoClearForces,
                "positionIterations": 3,  # Note: not in Box2D world but Step method, but here in RUBE JSON
                "velocityIterations": 8,  # Note: not in Box2D world but Step method, but here in RUBE JSON
                "stepsPerSecond": 60,  # Note: not in Box2D world but Step method, but here in RUBE JSON
                "warmStarting": obj.warmStarting,
                "continuousPhysics": obj.continuousPhysics,
                "subStepping": obj.subStepping,
                "body": [],
                "image": [],  # "image" attribute is left empty, since Box2D itself does not manage images.
                "joint": []
            }
            for body in obj.bodies:
                body_json = self.body_to_json(body)
                if body_json:
                    world_json["body"].append(body_json)

            for joint in obj.joints:
                joint_json = self.joint_to_json(joint)
                if joint_json:
                    world_json["joint"].append(joint_json)

            return world_json

        # Call the default method for other types
        return json.JSONEncoder.default(self, obj)

    def body_to_json(self, body):
        # Set basic body attributes
        body_json = {
            "allowSleep": body.sleepingAllowed,  # Not in RUBE JSON, but in Box2D
            "angle": body.angle,
            "angularDamping": body.angularDamping,
            "angularVelocity": body.angularVelocity,
            "awake": body.awake,
            "bullet": body.bullet,
            "enabled": body.active,  # Note: may become 'body.enabled' in future versions of pybox2d
            "fixedRotation": body.fixedRotation,
            "gravityScale": body.gravityScale,
            "massData-mass": body.massData.mass,  # == body.mass
            "massData-center": self.b2_vec2_to_rube_vec(body.massData.center),  # == body.localCenter
            "massData-I": body.massData.I,  # == body.inertia
            "linearDamping": body.linearDamping,
            "linearVelocity": self.b2_vec2_to_rube_vec(body.linearVelocity),
            "position": self.b2_vec2_to_rube_vec(body.position),  # == body.worldCenter
            "type": body.type,
            "fixture": []
        }
        # TODO Check that type corresponds to 0 = static, 1 = kinematic, 2 = dynamic

        # Add a name/id as "name" property for RUBE Json, since it does not exist in Box2D Body object
        if isinstance(body.userData, dict):
            if "id" in body.userData:
                body_json["name"] = body.userData["id"]
            elif "name" in body.userData:
                body_json["name"] = body.userData["name"]

        # Manage UserData if and only if it is serializable
        try:
            json.dumps(body.userData)
            body_json["customProperties"] = body.userData
        except (TypeError, OverflowError):
            print('Body user data could not be serialized.')

        # Add Fixtures
        for fixture in body.fixtures:
            fixture_json = self.fixture_to_json(fixture)
            if fixture_json:
                body_json["fixture"].append(fixture_json)

        return body_json

    def fixture_to_json(self, fixture):
        fixture_json = {
            "density": fixture.density,
            "filter-categoryBits": fixture.filterData.categoryBits,
            "filter-maskBits": fixture.filterData.maskBits,
            "filter-groupIndex": fixture.filterData.groupIndex,
            "friction": fixture.friction,
            "restitution": fixture.restitution,
            "sensor": fixture.sensor
        }

        if isinstance(fixture.shape, b2.b2PolygonShape):
            fixture_json["polygon"] = {
                "vertices": self.b2_vec2_arr_to_rube_vec_arr(fixture.shape.vertices)
            }
        elif isinstance(fixture.shape, b2.b2CircleShape):
            fixture_json["circle"] = {
                "center": self.b2_vec2_to_rube_vec(fixture.shape.pos),
                "radius": fixture.shape.radius
            }
        elif isinstance(fixture.shape, b2.b2ChainShape):
            fixture_json["polygon"] = {
                "vertices": self.b2_vec2_arr_to_rube_vec_arr(fixture.shape.vertices),
                "hasNextVertex": fixture.shape.m_hasNextVertex,
                "hasPrevVertex": fixture.shape.m_hasPrevVertex,
                "nextVertex": self.b2_vec2_to_rube_vec(fixture.shape.m_nextVertex),
                "prevVertex": self.b2_vec2_to_rube_vec(fixture.shape.m_prevVertex)
            }
        elif isinstance(fixture.shape, b2.b2EdgeShape):
            fixture_json["polygon"] = {
                "vertices": self.b2_vec2_arr_to_rube_vec_arr(fixture.shape.vertices)
            }
        else:
            print("Fixture shape is invalid")
            return None

        # Add a name/id as "name" property for RUBE Json, since it does not exist in Box2D Fixture object
        if isinstance(fixture.userData, dict):
            if "id" in fixture.userData:
                fixture_json["name"] = fixture.userData["id"]
            elif "name" in fixture.userData:
                fixture_json["name"] = fixture.userData["name"]

        # Manage UserData if and only if it is serializable
        try:
            json.dumps(fixture.userData)
            fixture_json["customProperties"] = fixture.userData
        except (TypeError, OverflowError):
            print('Fixture user data could not be serialized.')

        return fixture_json

    def joint_to_json(self, joint):
        joint_json = {
            "bodyA": joint.bodyA,
            "bodyB": joint.bodyB,
            "anchorA": joint.anchorA,
            "anchorB": joint.anchorB,
            "collideConnected": joint.collideConnected
        }

        # Add a name/id as "name" property for RUBE Json, since it does not exist in Box2D Joint object
        if isinstance(joint.userData, dict):
            if "id" in joint.userData:
                joint_json["name"] = joint.userData["id"]
            elif "name" in joint.userData:
                joint_json["name"] = joint.userData["name"]

        # Manage UserData if and only if it is serializable
        try:
            json.dumps(joint.userData)
            joint_json["customProperties"] = joint.userData
        except (TypeError, OverflowError):
            print('Fixture user data could not be serialized.')

        # ---------------------------------------------------
        if isinstance(joint, b2.b2RevoluteJoint):
            joint_json["type"] = "revolute"
            # joint_json["angle"] = joint.angle  # Is in Box2D but not in RUBE json
            joint_json["enableLimit"] = joint.limitEnabled
            joint_json["enableMotor"] = joint.motorEnabled
            joint_json["jointSpeed"] = joint.speed
            joint_json["lowerLimit"] = joint.lowerLimit
            joint_json["maxMotorTorque"] = joint.maxMotorTorque  # May require a pull request to fix the write-only...
            joint_json["motorSpeed"] = joint.motorSpeed
            joint_json["refAngle"] = joint.GetReferenceAngle()
            joint_json["upperLimit"] = joint.upperLimit
        # ---------------------------------------------------
        elif isinstance(joint, b2.b2DistanceJoint):
            joint_json["type"] = "distance"
            joint_json["dampingRatio"] = joint.dampingRatio
            joint_json["frequency"] = joint.frequency
            joint_json["length"] = joint.length
        # ---------------------------------------------------
        elif isinstance(joint, b2.b2PrismaticJoint):
            joint_json["type"] = "prismatic"
            joint_json["enableLimit"] = joint.limitEnabled
            joint_json["enableMotor"] = joint.motorEnabled
            joint_json["jointSpeed"] = joint.speed
            joint_json["lowerLimit"] = joint.lowerLimit
            joint_json["maxMotorForce"] = joint.maxMotorForce
            joint_json["motorSpeed"] = joint.motorSpeed
            joint_json["refAngle"] = joint.GetReferenceAngle()
            joint_json["upperLimit"] = joint.upperLimit
            joint_json["localAxisA"] = self.b2_vec2_to_rube_vec(joint.GetLocalAxisA())
            # joint_json["translation"] = joint.translation  # Is in Box2D but not in RUBE json
        # ---------------------------------------------------
        elif isinstance(joint, b2.b2WheelJoint):
            joint_json["type"] = "wheel"
            joint_json["enableMotor"] = joint.motorEnabled
            joint_json["localAxisA"] = self.b2_vec2_to_rube_vec(joint.GetLocalAxisA())
            joint_json["maxMotorTorque"] = joint.maxMotorTorque
            joint_json["motorSpeed"] = joint.motorSpeed
            joint_json["springDampingRatio"] = joint.springDampingRatio
            joint_json["springFrequency"] = joint.springFrequencyHz
            # joint_json["translation"] = joint.translation  # Is in Box2D but not in RUBE json
            # joint_json["jointSpeed"] = joint.speed  # Is in Box2D but not in RUBE json
        # ---------------------------------------------------
        elif isinstance(joint, b2.b2RopeJoint):
            # Note: Expect change here to b2Rope, b2RopeDef, b2RopeTuning ? in future versions of pybox2d
            joint_json["type"] = "rope"
            joint_json["maxLength"] = joint.maxLength
        # ---------------------------------------------------
        elif isinstance(joint, b2.b2MotorJoint):
            joint_json["type"] = "motor"
            joint_json["maxForce"] = joint.maxForce
            joint_json["maxTorque"] = joint.maxTorque
            joint_json["angularOffset"] = joint.angularOffset
            joint_json["linearOffset"] = self.b2_vec2_arr_to_rube_vec_arr(joint.linearOffset)
            # joint_json["correctionFactor"] = joint.GetCorrectionFactor()  # TODO Make pull request to pybox2d to add
        # ---------------------------------------------------
        elif isinstance(joint, b2.b2WeldJoint):
            joint_json["type"] = "weld"
            joint_json["dampingRatio"] = joint.GetDampingRatio()
            joint_json["frequency"] = joint.GetFrequency()
            joint_json["refAngle"] = joint.GetReferenceAngle()
        # ---------------------------------------------------
        elif isinstance(joint, b2.b2FrictionJoint):
            joint_json["type"] = "friction"
            joint_json["maxForce"] = joint.maxForce
            joint_json["maxTorque"] = joint.maxTorque
        # ---------------------------------------------------
        elif isinstance(joint, b2.b2GearJoint):
            # Is in Box2D but not in RUBE json
            joint_json["type"] = "gear"
            joint_json["ratio"] = joint.ratio
        # ---------------------------------------------------
        elif isinstance(joint, b2.b2MouseJoint):
            # Is in Box2D but not in RUBE json
            joint_json["type"] = "mouse"
            joint_json["maxForce"] = joint.maxForce
            joint_json["dampingRatio"] = joint.dampingRatio
            joint_json["frequency"] = joint.frequency
        # ---------------------------------------------------
        elif isinstance(joint, b2.b2PulleyJoint):
            # Is in Box2D but not in RUBE json
            joint_json["type"] = "pulley"
            joint_json["ratio"] = joint.ratio
            joint_json["lengthA"] = joint.lengthA
            joint_json["lengthB"] = joint.lengthB
            joint_json["maxLengthA"] = joint.length1
            joint_json["maxLengthB"] = joint.length2
            joint_json["groundAnchorA"] = self.b2_vec2_to_rube_vec(joint.groundAnchorA)
            joint_json["groundAnchorB"] = self.b2_vec2_to_rube_vec(joint.groundAnchorB)
        else:
            joint_def = None
            print ("Unsupported joint type")

        return joint_json

    def b2_vec2_to_rube_vec(self, b2_vec2):
        return {"x": b2_vec2.x, "y": b2_vec2.y}

    def b2_vec2_arr_to_rube_vec_arr(self, b2_vec2_arr):
        return {"x": [b2_vec2.x for b2_vec2 in b2_vec2_arr], "y": [b2_vec2.y for b2_vec2 in b2_vec2_arr]}


class B2WorldDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, world_json):
        b2_world = b2.b2World()

        if "gravity" in world_json:
            self.try_set_attr(world_json, "allowSleep", b2_world, "allowSleeping")
            self.try_set_attr(world_json, "autoClearForces", b2_world)
            self.try_set_attr(world_json, "continuousPhysics", b2_world)
            self.try_set_b2_vec2_attr(world_json, "gravity", b2_world)
            self.try_set_attr(world_json, "subStepping", b2_world)
            self.try_set_attr(world_json, "warmStarting", b2_world)

            # fill world with bodies and joints
            if "body" in world_json:
                for body_json in world_json["body"]:
                    self.add_body(b2_world, body_json)
            if "joint" in world_json:
                for joint_json in world_json["joint"]:
                    joint_def = self.create_joint_def(joint_json, b2_world)
                    if joint_def:
                        b2_world.CreateJoint(joint_def, joint_json["type"])

            return b2_world

        return world_json

    def add_body(self, b2_world, body_json):
        # Create body definition
        body_def = b2.b2BodyDef()
        self.try_set_attr(body_json, "allowSleep", body_def)
        self.try_set_attr(body_json, "angle", body_def)
        self.try_set_attr(body_json, "angularDamping", body_def)
        self.try_set_attr(body_json, "angularVelocity", body_def)
        self.try_set_attr(body_json, "awake", body_def)
        self.try_set_attr(body_json, "bullet", body_def)
        self.try_set_attr(body_json, "enabled", body_def, "active")
        self.try_set_attr(body_json, "fixedRotation", body_def)
        self.try_set_attr(body_json, "linearDamping", body_def)
        self.try_set_b2_vec2_attr(body_json, "linearVelocity", body_def)
        self.try_set_b2_vec2_attr(body_json, "position", body_def)
        self.try_set_attr(body_json, "gravityScale", body_def)
        self.try_set_attr(body_json, "type", body_def)

        # Manage UserData
        if "customProperties" in body_json and isinstance(body_json["customProperties"], dict):
            body_def.userData = body_json["customProperties"]
            # Add a name/id as "name" property if there is not one already
            if "id" not in body_def.userData:
                body_def.userData["id"] = body_json["name"]
            if "name" not in body_def.userData:
                body_def.userData["name"] = body_json["name"]

        # Create body
        body = b2_world.CreateBody(body_def)

        # Add body fixtures
        for fixture in body_json["fixture"]:
            self.add_fixture(body, fixture)

        # If massData is overriden from fixture data, take it into account
        if "massData-mass" in body_json and "massData-center" in body_json and "massData-I" in body_json:
            body.massData.mass = body_json["massData-mass"]
            body.massData.center = body_json["massData-center"]
            body.massData.I = body_json["massData-I"]

    def create_joint_def(self, joint_json, b2_world):
        joint_type = joint_json["type"]

        # ---------------------------------------------------
        if joint_type == "revolute":
            joint_def = b2.b2RevoluteJointDef()

            self.try_set_attr(joint_json, "enableLimit", joint_def)
            self.try_set_attr(joint_json, "enableMotor", joint_def)
            self.try_set_attr(joint_json, "jointSpeed", joint_def, "motorSpeed")
            self.try_set_attr(joint_json, "lowerLimit", joint_def, "lowerAngle")
            self.try_set_attr(joint_json, "maxMotorTorque", joint_def)
            self.try_set_attr(joint_json, "motorSpeed", joint_def)
            self.try_set_attr(joint_json, "refAngle", joint_def, "referenceAngle")
            self.try_set_attr(joint_json, "upperLimit", joint_def, "upperAngle")
        # ---------------------------------------------------
        elif joint_type == "distance":
            joint_def = b2.b2DistanceJointDef()

            self.try_set_attr(joint_json, "dampingRatio", joint_def)
            self.try_set_attr(joint_json, "frequency", joint_def, "frequencyHz")
            self.try_set_attr(joint_json, "length", joint_def)
        # ---------------------------------------------------
        elif joint_type == "prismatic":
            joint_def = b2.b2PrismaticJointDef()
            self.try_set_attr(joint_json, "enableLimit", joint_def)
            self.try_set_attr(joint_json, "enableMotor", joint_def)
            self.try_set_b2_vec2_attr(joint_json, "localAxisA", joint_def, "axis")
            self.try_set_attr(joint_json, "lowerLimit", joint_def, "lowerTranslation")
            self.try_set_attr(joint_json, "maxMotorForce", joint_def)
            self.try_set_attr(joint_json, "motorSpeed", joint_def)
            self.try_set_attr(joint_json, "refAngle", joint_def, "referenceAngle")
            self.try_set_attr(joint_json, "upperLimit", joint_def, "upperTranslation")
        # ---------------------------------------------------
        elif joint_type == "wheel":
            joint_def = b2.b2WheelJointDef()

            self.try_set_attr(joint_json, "enableMotor", joint_def)
            self.try_set_b2_vec2_attr(joint_json, "localAxisA", joint_def)
            self.try_set_attr(joint_json, "maxMotorTorque", joint_def)
            self.try_set_attr(joint_json, "motorSpeed", joint_def)
            self.try_set_attr(joint_json, "springDampingRatio", joint_def, "dampingRatio")
            self.try_set_attr(joint_json, "springFrequency", joint_def, "frequencyHz")
        # ---------------------------------------------------
        elif joint_type == "rope":
            joint_def = b2.b2RopeJointDef()

            self.try_set_attr(joint_json, "maxLength", joint_def)
        # ---------------------------------------------------
        elif joint_type == "motor":
            joint_def = b2.b2MotorJointDef()

            self.try_set_attr(joint_json, "maxForce", joint_def)
            self.try_set_attr(joint_json, "maxTorque", joint_def)
            self.try_set_b2_vec2_attr(joint_json, "linearOffset", joint_def)
            self.try_set_attr(joint_json, "angularOffset", joint_def)
            self.try_set_attr(joint_json, "correctionFactor", joint_def)
        # ---------------------------------------------------
        elif joint_type == "weld":
            joint_def = b2.b2WeldJointDef()

            self.try_set_attr(joint_json, "refAngle", joint_def, "referenceAngle")
            self.try_set_attr(joint_json, "dampingRatio", joint_def)
            self.try_set_attr(joint_json, "frequency", joint_def, "frequencyHz")
        # ---------------------------------------------------
        elif joint_type == "friction":
            joint_def = b2.b2FrictionJointDef()

            self.try_set_attr(joint_json, "maxForce", joint_def)
            self.try_set_attr(joint_json, "maxTorque", joint_def)
        # ---------------------------------------------------
        elif joint_type == "gear":
            # Is in Box2D but not in RUBE json
            joint_def = b2.b2GearJointDef()
            self.try_set_attr(joint_json, "ratio", joint_def)
        # ---------------------------------------------------
        elif joint_type == "mouse":
            # Is in Box2D but not in RUBE json
            joint_def = b2.b2MouseJointDef()
            self.try_set_attr(joint_json, "maxForce", joint_def)
            self.try_set_attr(joint_json, "dampingRatio", joint_def)
            self.try_set_attr(joint_json, "frequency", joint_def, "frequencyHz")
        # ---------------------------------------------------
        elif joint_type == "pulley":
            # Is in Box2D but not in RUBE json
            joint_def = b2.b2PulleyJointDef()
            self.try_set_attr(joint_json, "ratio", joint_def)
            self.try_set_attr(joint_json, "lengthA", joint_def)
            self.try_set_attr(joint_json, "lengthB", joint_def)
            self.try_set_attr(joint_json, "maxLengthA", joint_def)
            self.try_set_attr(joint_json, "maxLengthB", joint_def)
            self.try_set_b2_vec2_attr(joint_json, "groundAnchorA", joint_def)
            self.try_set_b2_vec2_attr(joint_json, "groundAnchorB", joint_def)
        else:
            joint_def = None
            print ("unsupported joint type")

        joint_def.bodyA = self.get_body(b2_world, joint_json["bodyA"])
        joint_def.bodyB = self.get_body(b2_world, joint_json["bodyB"])
        self.try_set_b2_vec2_attr(joint_json, "anchorA", joint_def)
        self.try_set_b2_vec2_attr(joint_json, "anchorB", joint_def)
        self.try_set_attr(joint_json, "collideConnected", joint_def)

        # Manage UserData
        if "customProperties" in joint_json and isinstance(joint_json["customProperties"], dict):
            joint_def.userData = joint_json["customProperties"]
            # Add a name/id as "name" property if there is not one already
            if "id" not in joint_def.userData:
                joint_def.userData["id"] = joint_json["name"]
            if "name" not in joint_def.userData:
                joint_def.userData["name"] = joint_json["name"]

        return joint_def

    def get_body(self, b2_world, index):
        return b2_world.bodies[index]

    def add_fixture(self, b2_world_body, fixture_json):
        # create and fill fixture definition
        fixture_def = b2.b2FixtureDef()

        # Done with issues:
        # missing pybox2d "filter" b2BodyDef property

        # special case for rube documentation of
        # "filter-categoryBits": 1, //if not present, interpret as 1
        if "filter-categoryBits" in fixture_json.keys():
            self.try_set_attr(fixture_json, "filter-categoryBits", fixture_def, "categoryBits")
        else:
            fixture_def.categoryBits = 1

        # special case for Rube Json property
        # "filter-maskBits": 1, //if not present, interpret as 65535
        if "filter-maskBits" in fixture_json.keys():
            self.try_set_attr(fixture_json, "filter-maskBits", fixture_def, "maskBits")
        else:
            fixture_def.maskBits = 65535

        self.try_set_attr(fixture_json, "density", fixture_def)
        self.try_set_attr(fixture_json, "filter-groupIndex", fixture_def, "groupIndex")
        self.try_set_attr(fixture_json, "friction", fixture_def)
        self.try_set_attr(fixture_json, "sensor", fixture_def, "isSensor")
        self.try_set_attr(fixture_json, "restitution", fixture_def)

        # fixture has one shape that is
        # polygon, circle or chain in json
        # chain may be open or loop, or edge in pyBox2D
        if "circle" in fixture_json.keys():
            if fixture_json["circle"]["center"] == 0:
                center_b2_vec2 = b2.b2Vec2(0, 0)
            else:
                center_b2_vec2 = self.rube_vec_to_b2_vec2(fixture_json["circle"]["center"])
            fixture_def.shape = b2.b2CircleShape(pos=center_b2_vec2, radius=fixture_json["circle"]["radius"])

        if "polygon" in fixture_json.keys():  # works ok
            polygon_vertices = self.rube_vec_arr_to_b2_vec2_arr(fixture_json["polygon"]["vertices"])
            fixture_def.shape = b2.b2PolygonShape(vertices=polygon_vertices)

        if "chain" in fixture_json.keys():  # works ok
            chain_vertices = self.rube_vec_arr_to_b2_vec2_arr(fixture_json["chain"]["vertices"])

            if len(chain_vertices) >= 3:
                # closed-loop b2LoopShape
                if "hasNextVertex" in fixture_json["chain"].keys():

                    # del last vertice to prevent crash from first and last
                    # vertices being to close
                    del chain_vertices[-1]

                    fixture_def.shape = b2.b2LoopShape(vertices_loop=chain_vertices, count=len(chain_vertices))

                    self.try_set_attr(fixture_json["chain"], "hasNextVertex", fixture_def.shape, "m_hasNextVertex", )
                    self.try_set_b2_vec2_attr(fixture_json["chain"], "nextVertex", fixture_def, "m_nextVertex")
                    self.try_set_attr(fixture_json["chain"], "hasPrevVertex", fixture_def.shape, "m_hasPrevVertex")
                    self.try_set_b2_vec2_attr(fixture_json["chain"], "prevVertex", fixture_def.shape, "m_prevVertex")

                else:  # open-ended ChainShape
                    fixture_def.shape = b2.b2ChainShape(vertices_chain=chain_vertices, count=len(chain_vertices))

            # json chain is b2EdgeShape
            if len(chain_vertices) < 3:
                fixture_def.shape = b2.b2EdgeShape(vertices=chain_vertices)

        # Manage UserData
        if "customProperties" in fixture_json and isinstance(fixture_json["customProperties"], dict):
            fixture_def.userData = fixture_json["customProperties"]
            # Add a name/id as "name" property if there is not one already
            if "id" not in fixture_def.userData:
                fixture_def.userData["id"] = fixture_json["name"]
            if "name" not in fixture_def.userData:
                fixture_def.userData["name"] = fixture_json["name"]

        # Create fixture
        b2_world_body.CreateFixture(fixture_def)

    def rube_vec_to_b2_vec2(self, rube_vec):
        return b2.b2Vec2(rube_vec["x"], rube_vec["y"])

    def rube_vec_arr_to_b2_vec2_arr(self, vector_array):
        return [b2.b2Vec2(x, y) for x, y in zip(vector_array["x"], vector_array["y"])]

    def try_set_attr(self, source_dict, source_key, target_obj, target_attr=None):
        """
        Assigns values from dict to target object, if key exists in dict
        may take renamed attribute for object
        works only with built_in values
        :param source_dict:
        :type source_dict:
        :param source_key: dict_source's key
        :type source_key:
        :param target_obj: obj with attribute 'key' or 'renamed'
        :type target_obj:
        :param target_attr: target attribute == key if is None
        :type target_attr:
        :return:
        :rtype:
        """
        if source_key in source_dict:
            if not target_attr:
                target_attr = source_key
            if hasattr(target_obj, target_attr):
                setattr(target_obj, target_attr, source_dict[source_key])
            else:
                print("No attr: " + target_attr + " in object")
        # debug helper
        # else:
        #    print "No key '" + source_key + "' in dict '" + source_dict["name"] + "'"

    def try_set_b2_vec2_attr(self, source_dict, source_key, target_obj, target_attr=None):
        """

        :param source_dict:
        :type source_dict:
        :param source_key:
        :type source_key:
        :param target_obj:
        :type target_obj:
        :param target_attr:
        :type target_attr:
        :return:
        :rtype:
        """
        if source_key in source_dict:
            # setting attr name
            if target_attr is None:
                target_attr = source_key

            # preparing B2Vec
            if source_dict[source_key] == 0:
                vec2 = b2.b2Vec2(0, 0)
            else:
                vec2 = self.rube_vec_to_b2_vec2(source_dict[source_key])

            # setting obj's attr value
            setattr(target_obj, target_attr, vec2)
        # else:
        #    print "No key '" + key + "' in dict '" + dict_source["name"] + "'"


if __name__ == "__main__":
    # 1. Empty world test
    empty_world_filepath = "empty_world_test.json"

    ## Dump
    empty_world = b2.b2World()
    with open(empty_world_filepath, "w") as f:
        json.dump(empty_world, f, cls=B2WorldEncoder)

    ## Load
    with open(empty_world_filepath, "r") as f:
        loaded_empty_world = json.load(f, cls=B2WorldDecoder)
        print(loaded_empty_world)

    ## Cleanup
    import os
    os.remove(empty_world_filepath)
