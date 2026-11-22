#include "imgui_viewer.h"
#include "budget_viewer_client.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>

ImGuiViewer::ImGuiViewer(std::shared_ptr<BudgetViewerClient> client)
    : budget_client_(std::move(client)), training_(false),
      glfw_window_width_(1600), glfw_window_height_(900), panel_width_(400) {
    const auto& calibration = budget_client_->metadata()["calibration"];
    image_width_ = calibration["width"].asInt();
    image_height_ = calibration["height"].asInt();
    main_fx_ = calibration["fx"].asFloat();
    main_fy_ = calibration["fy"].asFloat();
    main_cx_ = image_width_ * 0.5f;
    main_cy_ = image_height_ * 0.5f;
    tracking_vision_ = false;
}

void ImGuiViewer::runBudget() {
    // 原Viewer的GLFW/ImGui/OpenGL显示及handleUserInput相机控制；推理后端只读。
    glfwSetErrorCallback([](int error, const char* description) {
        std::cerr << "[view_budget] GLFW " << error << ": " << description << std::endl;
    });
    if (!glfwInit()) throw std::runtime_error("Cannot initialize original Viewer GLFW display");
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    GLFWwindow* window = glfwCreateWindow(glfw_window_width_, glfw_window_height_,
        "CaRtGS - Fixed Point Budget", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Cannot create original Viewer OpenGL window");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    std::cout << "NATIVE_OPENGL vendor=" << glGetString(GL_VENDOR)
              << " renderer=" << glGetString(GL_RENDERER)
              << " version=" << glGetString(GL_VERSION) << std::endl;
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsClassic();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");
    GLuint textures[2];
    glGenTextures(2, textures);
    for (const auto texture : textures) {
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    const auto& meta = budget_client_->metadata();
    const auto& frames = meta["frames"];
    int frame_index = 0, budget = meta["default_budget"].asInt();
    const int total = meta["total_gaussians"].asInt();
    bool free_camera = false, play = false, reference_full = false, profile_gl = false;
    int view_mode = 0, budget_unit = 0, training_ratio_index = 0, ablation_group = 0;
    bool view_conditioned=meta["view_residual_available"].asBool();
    double ratio_budget = .4;
    int lod_sorting = 0;
    double lod_ratios[] = {.5, .3, .1}, lod_shares[] = {.2, .3, .5};
    bool lod_colorize = false, lod_hysteresis = false, lod_reset_history = false;
    bool lod_mixed = false;
    int lod_mixed_seed = meta["mixed_ratio_defaults"].get("seed",0).asInt();
    unsigned int lod_mixed_sample = 0;
    double lod_equivalent_ratio = .3;
    bool dirty = true, have_image = false, have_reference = false;
    std::string error;
    std::string reference_cache_key;
    Json::Value stats;
    bool anchor_levels[] = {true, true, true};
    double last_play = 0;
    auto reset_camera = [&]() {
        Eigen::Matrix4f pose;
        const auto& values = frames[frame_index]["pose"];
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) pose(r, c) = values[r][c].asFloat();
        Twc_main_ = Sophus::SE3f(Eigen::Quaternionf(pose.block<3, 3>(0, 0)).normalized(),
                                pose.block<3, 1>(0, 3));
        Tcw_main_ = Twc_main_.inverse();
        free_camera = false;
        reset_main_to_init_ = false;
        dirty = true;
    };
    reset_camera();
    std::cout << "NATIVE_VIEWER_READY frames=" << frames.size() << " budget=" << budget << std::endl;
    while (!isStopped() && !glfwWindowShouldClose(window)) {
        const auto viewer_started = std::chrono::steady_clock::now();
        bool rendered_this_frame = false;
        glfwPollEvents();
        const auto after_poll = std::chrono::steady_clock::now();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        const auto after_begin = std::chrono::steady_clock::now();
        bool camera_input = !io.WantCaptureMouse &&
            (io.MouseWheel != 0 || ((io.MouseDown[0] || io.MouseDown[1] || io.MouseDown[2]) &&
                                   (io.MouseDelta.x != 0 || io.MouseDelta.y != 0)));
        if (!io.WantCaptureKeyboard) {
            for (auto key : {ImGuiKey_W, ImGuiKey_S, ImGuiKey_A, ImGuiKey_D, ImGuiKey_I,
                             ImGuiKey_K, ImGuiKey_J, ImGuiKey_L, ImGuiKey_U, ImGuiKey_O})
                camera_input |= ImGui::IsKeyDown(key);
        }
        if (camera_input) {
            handleUserInput();
            free_camera = true;
            play = false;
            dirty = true;
            if (reset_main_to_init_) reset_camera();
        }
        if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_R)) {
            play = false;
            reset_camera();
        }
        if (play && glfwGetTime() - last_play >= 0.1) {
            frame_index = (frame_index + 1) % frames.size();
            last_play = glfwGetTime();
            reset_camera();
        }
        ImGui::SetNextWindowPos(ImVec2(glfw_window_width_ - panel_width_, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(panel_width_, glfw_window_height_), ImGuiCond_Always);
        ImGui::Begin("Fixed Point Budget", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
        dirty |= ImGui::RadioButton("Budget", &view_mode, 0); ImGui::SameLine();
        dirty |= ImGui::RadioButton("Global ratio", &view_mode, 1); ImGui::SameLine();
        dirty |= ImGui::RadioButton("Full", &view_mode, 2);
        dirty |= ImGui::RadioButton("Ranked Anchor LoD", &view_mode, 3);
        if(meta["view_residual_available"].asBool()&&(view_mode==0||view_mode==3))
            dirty |= ImGui::Checkbox(meta["view_conditioning_type"].asString()=="shared_input"?
                "View-conditioned Selector":"View residual selection",&view_conditioned);
        if (ImGui::CollapsingHeader("Anchor levels", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (int i = 0; i < 3; ++i) {
                const auto& entry = meta["anchor_levels"][i];
                char label[128];std::snprintf(label, sizeof(label), "Anchor %d | %d anchors | %d GS",
                    i, entry["anchors"].asInt(), entry["gaussians"].asInt());
                dirty |= ImGui::Checkbox(label, &anchor_levels[i]);
                if (stats.isMember("anchor_levels")) {
                    const auto& level = stats["anchor_levels"][i];
                    ImGui::Text("  In view: %d anchors / %d GS; submitted: %d GS",
                        level["visible_anchors"].asInt(), level["visible_gaussians"].asInt(),
                        level["submitted_gaussians"].asInt());
                }
            }
        }
        if (view_mode != 2) {
            ImGui::TextUnformatted("Ablation:"); ImGui::SameLine();
            for (int i = 0; i < 4; ++i) {
                const char* labels[] = {"A", "B", "C", "D"};
                const char* help[] = {"Network + compensation", "Network / no compensation",
                                      "Random + compensation", "Random / no compensation"};
                if (i) ImGui::SameLine();
                dirty |= ImGui::RadioButton(labels[i], &ablation_group, i);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", help[i]);
            }
        }
        if (view_mode == 0) {
        dirty |= ImGui::RadioButton("Point count", &budget_unit, 0); ImGui::SameLine();
        dirty |= ImGui::RadioButton("Ratio budget", &budget_unit, 1);
        if (budget_unit == 0) {
        if (ImGui::InputInt("Point budget", &budget, 100, 1000)) {
            budget = std::max(0, budget);
            dirty = true;
        }
        dirty |= ImGui::SliderInt("Budget slider", &budget, 0, std::max(total, 1));
        } else {
            if (ImGui::InputDouble("Budget ratio", &ratio_budget, .01, .1, "%.4f")) {
                ratio_budget = std::clamp(ratio_budget, 0., 1.);
                dirty = true;
            }
            ImGui::TextUnformatted("K = floor(r * visible candidate N)");
        }
        } else if (view_mode == 1) {
            const auto& knots = meta["trained_ratios"];
            for (int i = 0; i < int(knots.size()); ++i) {
                if (i && i % 4) ImGui::SameLine();
                char text[32];std::snprintf(text, sizeof(text), "%.2f%%", 100 * knots[i].asDouble());
                dirty |= ImGui::RadioButton(text, &training_ratio_index, i);
            }
            ImGui::TextWrapped(ablation_group < 2 ? "Whole map; no frustum filter. Original FlexGS inference percentile."
                                                  : "Whole map random selection; K matches Network at this ratio.");
        } else if (view_mode == 3) {
            dirty |= ImGui::Checkbox("Gaussian mixed ratio", &lod_mixed);
            if (lod_mixed) {
                if (ImGui::InputDouble("Equivalent ratio", &lod_equivalent_ratio, .01, .1, "%.4f")) {
                    lod_equivalent_ratio = std::clamp(lod_equivalent_ratio,
                        meta["trained_ratios"][0].asDouble(),meta["trained_ratios"][meta["trained_ratios"].size()-1].asDouble());
                    dirty = true;
                }
                if (ImGui::InputInt("Mixed seed", &lod_mixed_seed)) {lod_mixed_seed = std::max(0,lod_mixed_seed);dirty = true;}
                if (ImGui::Button("Resample mixed ratios")) {++lod_mixed_sample;dirty = true;}
                ImGui::Text("Sample: %u",lod_mixed_sample);
                ImGui::TextWrapped("Longest projected side; Gaussian-weighted groups; fixed seed/sample at unchanged pose.");
            } else {
            dirty |= ImGui::RadioButton("Projected size", &lod_sorting, 0);ImGui::SameLine();
            dirty |= ImGui::RadioButton("Camera depth", &lod_sorting, 1);
            ImGui::TextWrapped(lod_sorting == 0 ? "Nominal cell projection: largest first."
                                              : "Anchor center in camera space: smallest Z first.");
            for (int i = 0; i < 3; ++i) {
                const char* names[] = {"High", "Middle", "Low"};
                ImGui::PushID(i);
                ImGui::TextUnformatted(names[i]);
                if (ImGui::InputDouble("Ratio", &lod_ratios[i], .01, .1, "%.3f")) {
                    lod_ratios[i] = std::clamp(lod_ratios[i], .001, 1.);dirty = true;
                }
                ImGui::PopID();
            }
            if (ImGui::InputDouble("High share", &lod_shares[0], .01, .1, "%.3f")) {
                lod_shares[0] = std::clamp(lod_shares[0], 0., 1.);
                lod_shares[1] = std::min(lod_shares[1], 1.-lod_shares[0]);dirty = true;
            }
            if (ImGui::InputDouble("Middle share", &lod_shares[1], .01, .1, "%.3f")) {
                lod_shares[1] = std::clamp(lod_shares[1], 0., 1.-lod_shares[0]);dirty = true;
            }
            lod_shares[2] = 1.-lod_shares[0]-lod_shares[1];
            ImGui::Text("Low share: %.1f%% (Gaussian count)", 100*lod_shares[2]);
            if (!(lod_ratios[0] > lod_ratios[1] && lod_ratios[1] > lod_ratios[2]))
                ImGui::TextWrapped("Ratios must satisfy high > middle > low > 0.");
            dirty |= ImGui::Checkbox("Rank hysteresis (2 percentage points)", &lod_hysteresis);
            }
            dirty |= ImGui::Checkbox("Color levels (PSNR uses normal image)", &lod_colorize);
            if (ImGui::Button("Reset LoD history")) {lod_reset_history = true;dirty = true;}
        }
        dirty |= ImGui::Checkbox("Compare with Full at this pose", &reference_full);
        ImGui::Separator();
        if (ImGui::SliderInt("Dataset frame", &frame_index, 0, frames.size() - 1)) {
            play = false;
            reset_camera();
        }
        if (ImGui::Button("Previous")) {
            frame_index = (frame_index + frames.size() - 1) % frames.size(); reset_camera();
        }
        ImGui::SameLine();
        if (ImGui::Button("Next")) { frame_index = (frame_index + 1) % frames.size(); reset_camera(); }
        ImGui::SameLine();
        if (ImGui::Button("Reset pose (R)")) reset_camera();
        ImGui::Checkbox("Play trajectory (10 Hz max)", &play);
        if (ImGui::Button("Refresh / retry")) { dirty = true; error.clear(); }
        ImGui::TextUnformatted(free_camera ? "Free camera - no RGB ground truth" : "Saved dataset camera");
        ImGui::TextWrapped("%s", frames[frame_index]["name"].asCString());
        ImGui::Separator();
        if (!stats.empty()) {
            auto metric = [&](const char* key) {
                return stats[key].isNull() ? std::string("N/A") : stats[key].asString();
            };
            ImGui::Text("Visible anchors: %s", metric("visible_anchors").c_str());
            ImGui::Text("Enabled anchors / members: %s / %s", metric("enabled_anchors").c_str(), metric("enabled_anchor_gaussians").c_str());
            ImGui::Text("Enabled in view: %s anchors / %s GS", metric("enabled_visible_anchors").c_str(), metric("enabled_visible_gaussians").c_str());
            ImGui::Text("Submitted / projected anchors: %s / %s", metric("submitted_anchors").c_str(), metric("projected_anchors").c_str());
            ImGui::Text("Candidate N: %s", metric("candidate_gaussians").c_str());
            ImGui::Text("Selection scope: %s", metric("selection_scope").c_str());
            ImGui::Text("Ablation: %s / %s", metric("ablation_group").c_str(), metric("selection").c_str());
            ImGui::Text("Eligible candidates: %s", metric("eligible_candidates").c_str());
            ImGui::Text("Actual submitted K: %s", metric("active_gaussians").c_str());
            ImGui::Text("Projected points (radii > 0): %s", metric("projected_gaussians").c_str());
            ImGui::Text("Query ratio: %s", metric("ratio").c_str());
            if (stats["mode"].asString() == "ranked_lod") {
                ImGui::Text("Retained: %.2f%%", 100*stats["actual_retention_ratio"].asDouble());
                ImGui::Text("Weighted target: %.3f%%", 100*stats["weighted_target_ratio"].asDouble());
                if(stats["mixed_ratio"].asBool())ImGui::Text("Equivalent target: %.3f%%",100*stats["target_equivalent_ratio"].asDouble());
                for (const auto& group : stats["lod_groups"])
                    ImGui::Text("r%.4f share%.1f%%: N=%d K=%d", group["ratio"].asDouble(),
                        100*group["actual_share"].asDouble(), group["candidate_gaussians"].asInt(), group["active_gaussians"].asInt());
                if (stats["lod_colorize"].asBool())
                    ImGui::Text("Color: red/high green/middle blue/low; +%.2f ms",
                                stats["lod_debug_render_ms"].asDouble());
            }
            if (stats["ground_truth_psnr"].isNumeric()) ImGui::Text("PSNR vs RGB: %.3f dB", stats["ground_truth_psnr"].asDouble());
            else ImGui::Text("PSNR vs RGB: %s", metric("ground_truth_psnr").c_str());
            ImGui::Text("PSNR vs Full: %s dB", metric("full_reference_psnr").c_str());
            ImGui::Text("Prepare: %.2f ms / Raster: %.2f ms", stats["prepare_ms"].asDouble(), stats["rasterizer_ms"].asDouble());
            ImGui::Text("Prepare + raster: %.2f ms", stats["prepare_plus_render_ms"].asDouble());
            ImGui::Text("Backend roundtrip: %.2f ms", stats["backend_roundtrip_ms"].asDouble());
            ImGui::Text("Texture upload: %.2f / swap: %.2f ms", stats["texture_upload_ms"].asDouble(), stats["swap_ms"].asDouble());
            ImGui::Text("Changed frame total: %.2f ms", stats["viewer_frame_ms"].asDouble());
            if (stats["cache_hit"].asBool()) ImGui::TextUnformatted("View-independent data cached");
            if (!stats["ratio_in_training_range"].asBool())
                ImGui::TextWrapped("Ratio outside trained range. Quality is unvalidated.");
            if (stats["budget_shortfall"].asInt() > 0)
                ImGui::Text("Budget shortfall: %d", stats["budget_shortfall"].asInt());
        }
        if (!error.empty()) ImGui::TextWrapped("Backend error: %s", error.c_str());
        ImGui::Separator();
        ImGui::TextWrapped("Original CaRtGS controls: left drag = rotate; right drag = roll; middle drag = pan; wheel / WASD = move; IJKLUO = rotate.");
        ImGui::SliderFloat("Move speed", &keyboard_velocity_, 0.001f, 0.5f, "%.3f");
        if (ImGui::Checkbox("Measure GL completion (profiling)", &profile_gl)) dirty = true;
        if (have_reference) {
            ImGui::TextUnformatted("Reference: matching RGB / optional Full");
            const float width = panel_width_ - 24;
            ImGui::Image(reinterpret_cast<void*>(intptr_t(textures[1])), ImVec2(width, width * image_height_ / image_width_));
        }
        ImGui::End();
        const auto after_ui = std::chrono::steady_clock::now();
        if (dirty) {
            Json::Value request;
            request["budget"] = budget;
            request["mode"] = view_mode == 3 ? "ranked_lod" : view_mode == 2 ? "full" : view_mode == 1 ? "global_ratio" : "budget";
            request["budget_unit"] = budget_unit == 0 ? "points" : "ratio";
            request["ratio"] = view_mode == 1 ? meta["trained_ratios"][training_ratio_index].asDouble() : ratio_budget;
            request["reference_full"] = reference_full;
            request["selection"] = ablation_group < 2 ? "learned" : "random";
            request["compensation"] = ablation_group % 2 == 0;
            request["view_conditioned"]=view_conditioned&&(view_mode==0||view_mode==3);
            request["reference_cache_key"] = reference_cache_key;
            request["anchor_levels"] = Json::Value(Json::arrayValue);
            for (int i = 0; i < 3; ++i)
                if (anchor_levels[i]) request["anchor_levels"].append(i);
            if (view_mode == 3) {
                request["lod_sorting"] = lod_mixed || lod_sorting == 0 ? "projection" : "depth";
                request["lod_mixed"] = lod_mixed;
                request["lod_equivalent_ratio"] = lod_equivalent_ratio;
                request["lod_mixed_seed"] = lod_mixed_seed;
                request["lod_mixed_sample"] = lod_mixed_sample;
                for (int i = 0; i < 3; ++i) {
                    request["lod_ratios"][i] = lod_ratios[i];request["lod_shares"][i] = lod_shares[i];
                }
                request["lod_colorize"] = lod_colorize;
                request["lod_hysteresis"] = !lod_mixed && lod_hysteresis ? .02 : 0.;
                request["lod_reset_history"] = lod_reset_history;
            }
            if (free_camera) {
                const auto pose = Tcw_main_.inverse().matrix();
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c) request["pose"][r][c] = pose(r, c);
            } else request["frame"] = frame_index;
            try {
                const auto backend_started = std::chrono::steady_clock::now();
                const auto result = budget_client_->render(request);
                stats = result.metadata["stats"];
                stats["backend_roundtrip_ms"] = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - backend_started).count();
                const cv::Mat images[] = {result.image, result.reference};
                const auto upload_started = std::chrono::steady_clock::now();
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                for (int i = 0; i < 2; ++i) if (!images[i].empty()) {
                    glBindTexture(GL_TEXTURE_2D, textures[i]);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, images[i].cols, images[i].rows,
                                 0, GL_RGB, GL_FLOAT, images[i].data);
                }
                if (profile_gl) glFinish();
                stats["texture_upload_ms"] = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - upload_started).count();
                stats["profile_gl_completion"] = profile_gl;
                have_image = true;
                have_reference = result.metadata["reference_available"].asBool();
                reference_cache_key = result.metadata["reference_key"].isString() ? result.metadata["reference_key"].asString() : "";
                error.clear();
                lod_reset_history = false;
                rendered_this_frame = true;
            } catch (const std::exception& failure) {
                error = failure.what();
                play = false;
                std::cerr << "[view_budget] " << error << std::endl;
            }
            dirty = false;
        }
        const auto after_backend = std::chrono::steady_clock::now();
        const float canvas_width = glfw_window_width_ - panel_width_;
        const float scale = std::min(canvas_width / image_width_, float(glfw_window_height_) / image_height_);
        const ImVec2 origin((canvas_width - scale * image_width_) / 2,
                            (glfw_window_height_ - scale * image_height_) / 2);
        if (have_image) ImGui::GetBackgroundDrawList()->AddImage(reinterpret_cast<void*>(intptr_t(textures[0])),
            origin, ImVec2(origin.x + scale * image_width_, origin.y + scale * image_height_));
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui::Render();
        const auto draw_started = std::chrono::steady_clock::now();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        const auto swap_started = std::chrono::steady_clock::now();
        glfwSwapBuffers(window);
        if (rendered_this_frame) {
            stats["window_poll_ms"] = std::chrono::duration<double, std::milli>(after_poll - viewer_started).count();
            stats["imgui_begin_ms"] = std::chrono::duration<double, std::milli>(after_begin - after_poll).count();
            stats["draw_submit_ms"] = std::chrono::duration<double, std::milli>(swap_started - draw_started).count();
            stats["ui_build_ms"] = std::chrono::duration<double, std::milli>(after_ui - after_begin).count();
            stats["draw_setup_ms"] = std::chrono::duration<double, std::milli>(draw_started - after_backend).count();
            stats["swap_ms"] = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - swap_started).count();
            stats["viewer_frame_ms"] = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - viewer_started).count();
            Json::StreamWriterBuilder writer;writer["indentation"] = "";
            std::cout << "BUDGET_FRAME " << Json::writeString(writer, stats) << std::endl;
        }
    }
    glDeleteTextures(2, textures);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
}
