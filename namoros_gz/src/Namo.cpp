/*
 * Copyright (C) 2021 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// We'll use a string and the ignmsg command below for a brief example.
// Remove these includes if your plugin doesn't need them.
#include <string>
#include <gz/common/Console.hh>

#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>
#include "gz/sim/components/CanonicalLink.hh"
#include "gz/sim/components/JointPosition.hh"
#include "gz/sim/components/JointVelocityCmd.hh"
#include "gz/sim/components/DetachableJoint.hh"
#include "gz/sim/components/Link.hh"
#include "gz/sim/components/Model.hh"
#include "gz/sim/components/Name.hh"
#include "gz/sim/components/ParentEntity.hh"
#include "gz/sim/components/Pose.hh"
#include "gz/sim/Link.hh"
#include "gz/sim/Util.hh"

// Don't forget to include the plugin's header.
#include "Namo.hh"

using namespace gz;
using namespace gz::sim;
using namespace systems;
using namespace namoros_gz;

/// \brief Velocity command.
struct Commands
{
  /// \brief Linear velocity.
  double lin;

  /// \brief Angular velocity.
  double ang;

  Commands() : lin(0.0), ang(0.0) {}
};

struct GrabReleaseCommand
{
  std::string robotName;
  std::string robotLink;
  std::string obstacleName;
  std::string obstacleLink;
};

class namoros_gz::NamoPrivate
{

public:
  void OnGrabCommand(const msgs::Param_V &_msg);
  void OnReleaseCommand(const msgs::Param_V &_msg);
  /// \brief Ignition communication node.
  transport::Node node;
  std::unique_ptr<GrabReleaseCommand> grab;
  std::unique_ptr<GrabReleaseCommand> release;
  /// \brief A mutex to protect the params command.
  std::mutex mutex;
};

Namo::Namo()
    : dataPtr(std::make_unique<NamoPrivate>())
{
}

void NamoPrivate::OnGrabCommand(const msgs::Param_V &_msg)
{
  // std::lock_guard<std::mutex> lock(this->mutex);
  ignmsg << "grab" << std::endl;

  auto params = _msg.param(0).params();

  for (auto it = params.begin(); it != params.end(); ++it)
  {
    ignmsg << "Key: " << it->first << ", Value: " << it->second.string_value() << std::endl;
    ignmsg << "Type: " << it->second.type() << std::endl;
  }

  grab = std::make_unique<GrabReleaseCommand>();
  grab->robotName = params["robot_name"].string_value();
  grab->robotLink = params["robot_link"].string_value();
  grab->obstacleName = params["obstacle_name"].string_value();
  grab->obstacleLink = params["obstacle_link"].string_value();
}

void NamoPrivate::OnReleaseCommand(const msgs::Param_V &_msg)
{
  // std::lock_guard<std::mutex> lock(this->mutex);

  ignmsg << "release" << std::endl;

  auto params = _msg.param(0).params();

  for (auto it = params.begin(); it != params.end(); ++it)
  {
    ignmsg << "Key: " << it->first << ", Value: " << it->second.string_value() << std::endl;
    ignmsg << "Type: " << it->second.type() << std::endl;
  }

  grab = nullptr;
  release = std::make_unique<GrabReleaseCommand>();
  release->robotName = params["robot_name"].string_value();
  release->robotLink = params["robot_link"].string_value();
  release->obstacleName = params["obstacle_name"].string_value();
  release->obstacleLink = params["obstacle_link"].string_value();
}

void Namo::Configure(const gz::sim::Entity &_entity,
                     const std::shared_ptr<const sdf::Element> &_sdf,
                     gz::sim::EntityComponentManager &_ecm,
                     gz::sim::EventManager & /*_eventMgr*/)
{
  ignmsg << "CONFIGURING" << std::endl;

  // Subscribe to commands
  std::vector<std::string> topics;
  if (_sdf->HasElement("topic"))
  {
    topics.push_back(_sdf->Get<std::string>("topic"));
  }
  topics.push_back("/namo_grab");
  auto topic = validTopic(topics);

  this->dataPtr->node.Subscribe(topic, &NamoPrivate::OnGrabCommand,
                                this->dataPtr.get());
  this->dataPtr->node.Subscribe("/namo_release", &NamoPrivate::OnReleaseCommand,
                                this->dataPtr.get());
}

void Namo::PreUpdate(const gz::sim::UpdateInfo &_info,
                     gz::sim::EntityComponentManager &_ecm)
{
  if (this->dataPtr.get()->grab != nullptr)
  {
    auto cmd = this->dataPtr.get()->grab.get();

    ignmsg << "GRABBING" << std::endl;

    if (this->robotToJoint.count(cmd->robotName) > 0)
    {
      ignmsg << "ALREADED GRABBED: " << this->robotToJoint[cmd->robotName] << std::endl;
      this->dataPtr.get()->grab = nullptr;
      return;
    }

    auto robot = _ecm.EntityByComponents(
        components::Model(), components::Name(cmd->robotName));
    std::unordered_set<Entity> robotChildren = _ecm.Descendants(robot);

    Entity robotLink;
    bool robotLinkFound = false;
    for (auto it = robotChildren.begin(); it != robotChildren.end(); ++it)
    {
      const Entity x = *it;
      auto name = _ecm.ComponentData<components::Name>(x);
      std::cout
          << "NAME: " << name.value_or("") << "ROBOT_LINK: " << cmd->robotLink << std::endl;
      if (name.value_or("").compare(cmd->robotLink) == 0)
      {
        robotLink = x;
        robotLinkFound = true;
      }
    }

    if (robotLinkFound == false)
    {
      throw std::runtime_error("Robot link not found");
    }

    auto box = _ecm.EntityByComponents(
        components::Model(), components::Name(cmd->obstacleName));
    auto boxLink = _ecm.EntityByComponents(
        components::Link(), components::ParentEntity(box),
        components::Name(cmd->obstacleLink));

    auto joint = _ecm.CreateEntity();
    auto jointComponent = _ecm.CreateComponent(
        joint,
        components::DetachableJoint({robotLink,
                                     boxLink, "fixed"}));

    ignmsg << "Created Joint" << std::endl;

    this->robotToJoint[cmd->robotName] = joint;

    this->dataPtr.get()->grab = nullptr;
  }

  // handle release
  if (this->dataPtr.get()->release != nullptr)
  {
    auto cmd = this->dataPtr.get()->release.get();

    if (this->robotToJoint.count(cmd->robotName) > 0)
    {
      ignmsg << "RELEASING: " << this->robotToJoint[cmd->robotName] << std::endl;
      _ecm.RequestRemoveEntity(this->robotToJoint[cmd->robotName]);
      this->robotToJoint.erase(cmd->robotName);
    }

    this->dataPtr.get()->release = nullptr;
  }
}

// Here we implement the PostUpdate function, which is called at every
// iteration.
void Namo::PostUpdate(const gz::sim::UpdateInfo &_info,
                      const gz::sim::EntityComponentManager &_ecm)
{
  // This is a simple example of how to get information from UpdateInfo.
  std::string msg = "Hello, world! Simulation is ";
  if (!_info.paused)
    msg += "not ";
  msg += "paused.";

  // Messages printed with ignmsg only show when running with verbosity 3 or
  // higher (i.e. ign gazebo -v 3)
  // ignmsg << msg << std::endl;
}

// This is required to register the plugin. Make sure the interfaces match
// what's in the header.
IGNITION_ADD_PLUGIN(
    namoros_gz::Namo,
    gz::sim::System,
    namoros_gz::Namo::ISystemPreUpdate,
    namoros_gz::Namo::ISystemPostUpdate,
    namoros_gz::Namo::ISystemConfigure)
