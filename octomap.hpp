/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

#pragma once

#include <fins/node.hpp>
#include <fins/agent/parameter_server.hpp>
#include <fins/utils/time.hpp>

// OctoMap
#include <octomap/octomap.h>
#include <octomap/OcTree.h>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

// Messages
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <mutex>

class OctomapNode : public fins::Node {
public:
    void define() override {
        set_name("OctomapNode");
        set_description("Octomap Node with Raycasting");
        set_category("Mapping");

        register_input<pcl::PointCloud<pcl::PointXYZI>::Ptr>("cloud", &OctomapNode::on_cloud);
        register_input<geometry_msgs::msg::TransformStamped>("transform", &OctomapNode::on_transform);
        register_output<nav_msgs::msg::OccupancyGrid>("projected_map");
        register_output<visualization_msgs::msg::MarkerArray>("markerarray");

        register_parameter<double>("resolution", &OctomapNode::on_resolution_changed, 0.1);
        register_parameter<double>("occupancy_min_z", &OctomapNode::on_occ_min_z_changed, 0.15);
        register_parameter<double>("occupancy_max_z", &OctomapNode::on_occ_max_z_changed, 2.0);
        register_parameter<double>("max_range", &OctomapNode::on_max_range_changed, 10.0);
    }

    void initialize() override {
        logger->info("Initializing Octomap Node...");

        init_octree();

        cloud_count_ = 0;
        has_transform_ = false;
        logger->info("Octomap initialized: res={}, range={}, z_range=[{}, {}]", 
                     resolution_, max_range_, occ_min_z_, occ_max_z_);
    }

    void run() override {}
    void pause() override {}
    void reset() override {
        std::lock_guard<std::mutex> lock(mtx_);
        octree_->clear();
        logger->info("Octomap reset.");
    }

    void on_transform(const geometry_msgs::msg::TransformStamped &msg) {
        latest_origin_.x() = msg.transform.translation.x;
        latest_origin_.y() = msg.transform.translation.y;
        latest_origin_.z() = msg.transform.translation.z;
        has_transform_ = true;
    }

    void on_cloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr &msg, fins::AcqTime acq_time) {
        if (!has_transform_ || msg->empty()) return;

        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::VoxelGrid<pcl::PointXYZI> sor;
        sor.setInputCloud(msg);
        sor.setLeafSize(resolution_, resolution_, resolution_);
        sor.filter(*filtered_cloud);

        {
            std::lock_guard<std::mutex> lock(mtx_);
            insert_cloud(filtered_cloud, latest_origin_);
            
            if (cloud_count_ % 100 == 0) octree_->prune();
        }

        cloud_count_++;
        if (cloud_count_ % 5 == 0) {
            publish_all(acq_time);
        }
    }

private:
    void init_octree() {
        octree_ = std::make_unique<octomap::OcTree>(resolution_);
        octree_->setProbHit(0.7);
        octree_->setProbMiss(0.4);
        octree_->setClampingThresMin(0.12);
        octree_->setClampingThresMax(0.97);
    }

    void on_resolution_changed(double v) {
        if (resolution_ == v) return;
        resolution_ = v;
        logger->info("Resolution changed to {}. Rebuilding octree...", resolution_);
        std::lock_guard<std::mutex> lock(mtx_);
        init_octree();
    }

    void on_occ_min_z_changed(double v) { occ_min_z_ = v; }
    void on_occ_max_z_changed(double v) { occ_max_z_ = v; }
    void on_max_range_changed(double v) { max_range_ = v; }

    void insert_cloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, const octomap::point3d& sensor_origin) {
        octomap::KeySet free_cells, occupied_cells;

        for (const auto& pt : cloud->points) {
            octomap::point3d point(pt.x, pt.y, pt.z);

            double dist = (point - sensor_origin).norm();
            if (max_range_ > 0.0 && dist > max_range_) {
                point = sensor_origin + (point - sensor_origin).normalized() * max_range_;
            }

            octomap::KeyRay ray;
            if (octree_->computeRayKeys(sensor_origin, point, ray)) {
                free_cells.insert(ray.begin(), ray.end());
            }

            if (max_range_ < 0.0 || dist <= max_range_) {
                octomap::OcTreeKey key;
                if (octree_->coordToKeyChecked(point, key)) {
                    occupied_cells.insert(key);
                }
            }
        }

        for (const auto& key : free_cells) {
            if (occupied_cells.find(key) == occupied_cells.end()) {
                octree_->updateNode(key, false);
            }
        }
        for (const auto& key : occupied_cells) {
            octree_->updateNode(key, true);
        }
    }

    void publish_all(const fins::AcqTime& stamp) {
        publish_2d_grid(stamp);
        publish_markers(stamp);
    }

    void publish_2d_grid(const fins::AcqTime& stamp) {
        if (octree_->size() <= 1) return;

        double min_x, min_y, min_z, max_x, max_y, max_z;
        octree_->getMetricMin(min_x, min_y, min_z);
        octree_->getMetricMax(max_x, max_y, max_z);

        octomap::OcTreeKey min_key = octree_->coordToKey(min_x, min_y, min_z);
        octomap::OcTreeKey max_key = octree_->coordToKey(max_x, max_y, max_z);

        nav_msgs::msg::OccupancyGrid grid;
        grid.header.frame_id = world_frame_;
        grid.info.resolution = resolution_;
        grid.info.width = (max_key[0] - min_key[0]) + 1;
        grid.info.height = (max_key[1] - min_key[1]) + 1;

        if (grid.info.width > 2000 || grid.info.height > 2000) return;

        grid.info.origin.position.x = octree_->keyToCoord(min_key[0]) - resolution_ * 0.5;
        grid.info.origin.position.y = octree_->keyToCoord(min_key[1]) - resolution_ * 0.5;
        grid.data.assign(grid.info.width * grid.info.height, -1);

        for (auto it = octree_->begin_leafs(), end = octree_->end_leafs(); it != end; ++it) {
            if (it.getZ() >= occ_min_z_ && it.getZ() <= occ_max_z_) {
                int x = it.getKey()[0] - min_key[0];
                int y = it.getKey()[1] - min_key[1];
                
                if (x >= 0 && x < (int)grid.info.width && y >= 0 && y < (int)grid.info.height) {
                    size_t idx = y * grid.info.width + x;
                    if (octree_->isNodeOccupied(*it)) {
                        grid.data[idx] = 100;
                    } else if (grid.data[idx] == -1) {
                        grid.data[idx] = 0;
                    }
                }
            }
        }
        send("projected_map", grid, stamp);
    }

    void publish_markers(const fins::AcqTime& stamp) {
        visualization_msgs::msg::MarkerArray occupied_nodes;
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = world_frame_;
        marker.ns = "octomap";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::CUBE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = resolution_;
        marker.scale.y = resolution_;
        marker.scale.z = resolution_;
        marker.color.r = 0.0; marker.color.g = 0.5; marker.color.b = 1.0; marker.color.a = 0.6;

        for (auto it = octree_->begin_leafs(), end = octree_->end_leafs(); it != end; ++it) {
            if (octree_->isNodeOccupied(*it)) {
                geometry_msgs::msg::Point p;
                p.x = it.getX(); p.y = it.getY(); p.z = it.getZ();
                marker.points.push_back(p);
            }
        }
        if (!marker.points.empty()) {
            occupied_nodes.markers.push_back(marker);
            send("markerarray", occupied_nodes, stamp);
        }
    }

    std::unique_ptr<octomap::OcTree> octree_;
    std::mutex mtx_;

    double resolution_;
    double occ_min_z_;
    double occ_max_z_;
    double max_range_;
    std::string world_frame_ = "map";

    uint64_t cloud_count_;
    octomap::point3d latest_origin_{0, 0, 0};
    bool has_transform_{false};
};

EXPORT_NODE(OctomapNode)