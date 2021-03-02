#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C++ version Copyright (c) 2006-2007 Erin Catto http://www.box2d.org
# Python version by Ken Lauer / sirkne at gmail dot com
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software
# in a product, an acknowledgment in the product documentation would be
# appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.

import copy
import Box2D
from Box2D.examples.framework import (Framework, Keys, main)
from Box2D import (b2EdgeShape, b2FixtureDef, b2PolygonShape, b2_dynamicBody,
                   b2_kinematicBody, b2_staticBody)


class CollisionPairsContactListener(Box2D.b2ContactListener):
    def __init__(self, **kwargs):
        Box2D.b2ContactListener.__init__(self, **kwargs)
        self._contacts = {}

    def BeginContact(self, contact):
        self._contacts[(contact.fixtureA.userData['uid'], contact.fixtureB.userData['uid'])] = contact
        print(self.get_collision_pairs())

    def EndContact(self, contact):
        del self._contacts[(contact.fixtureA.userData['uid'], contact.fixtureB.userData['uid'])]
        print(self.get_collision_pairs())

    def get_collision_pairs(self):
        return list(self._contacts.keys())


class OverlappingEntitiesUidsQueryCallback(Box2D.b2QueryCallback):
    def __init__(self):
        Box2D.b2QueryCallback.__init__(self)
        self.overlapping_uids = set()

    def ReportFixture(self, fixture):
        self.overlapping_uids.add(fixture.userData['uid'])
        return True # Continue the query by returning True


class BodyTypes(Framework):
    name = "Body Types"
    description = "Change body type keys: (d) dynamic, (s) static, (k) kinematic"
    speed = 3  # platform speed

    def __init__(self):
        self.contact_listener = CollisionPairsContactListener()

        super(BodyTypes, self).__init__()

        self.world.contactListener = self.contact_listener
        self.world.gravity = (0., 0.)

        # The ground
        ground = self.world.CreateStaticBody(
            fixtures=b2FixtureDef(shape=b2EdgeShape(vertices=[(-20, 0), (20, 0)]), isSensor=True, userData={"uid": "ground"})
        )

        # The attachment
        # self.attachment = self.world.CreateDynamicBody(
        #     position=(0, 3),
        #     fixtures=b2FixtureDef(
        #         shape=b2PolygonShape(box=(0.5, 2)), density=2.0, isSensor=True, userData={"uid": "attachment"}),
        # )

        # The platform
        self.platform = self.world.CreateDynamicBody(
            position=(0, 5),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=(4, 0.5)), isSensor=True, userData={"uid": "platform"}
            ),
            bullet=True
        )

        # The joints joining the attachment/platform and ground/platform
        # self.world.CreateWeldJoint(
        #     bodyA=self.attachment,
        #     bodyB=self.platform,
        #     anchor=(0, 5)
        # )

        # self.world.CreatePrismaticJoint(
        #     bodyA=ground,
        #     bodyB=self.platform,
        #     anchor=(0, 5),
        #     axis=(1, 0),
        #     maxMotorForce=1000,
        #     enableMotor=True,
        #     lowerTranslation=-10,
        #     upperTranslation=10,
        #     enableLimit=True
        # )

        # And the payload that initially sits upon the platform
        # Reusing the fixture we previously defined above
        self.payload = self.world.CreateDynamicBody(
            position=(0, 8),
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(box=(0.75, 0.75)), isSensor=True, userData={"uid": "payload"}
            ),
            bullet=True
        )


        self.world.CreateDistanceJoint(
            bodyA=self.payload,
            bodyB=self.platform,
            anchorA=(0, 8),
            anchorB=(0, 5),
            frequencyHz=0.,
            dampingRatio=0.
        )

        self.payload.linearVelocity = self.payload.linearVelocity[0], -1.5

        self.settings.hz = 1
        self.settings.positionIterations = 1
        self.settings.velocityIterations = 1

    def Keyboard(self, key):
        if key == Keys.K_w:
            self.payload.linearVelocity = 0., 0.
        elif key == Keys.K_s:
            self.payload.linearVelocity = self.payload.linearVelocity[0], -5.
        elif key == Keys.K_d:
            self.payload.linearVelocity = 5., self.payload.linearVelocity[1]
        elif key == Keys.K_k:
            self.platform.type = b2_kinematicBody
            self.platform.linearVelocity = (-self.speed, 0)
            self.platform.angularVelocity = 0

    def Step(self, settings):
        print("self.payload.position: {}, self.platform.position: {}".format(self.payload.position, self.platform.position))
        super(BodyTypes, self).Step(self.settings)

if __name__ == "__main__":
    main(BodyTypes)
