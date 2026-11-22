#include "budget_viewer_client.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace {
Json::Value parseJson(const char* begin, const char* end) {
  Json::CharReaderBuilder builder;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  Json::Value result;
  std::string error;
  if (!reader->parse(begin, end, &result, &error))
    throw std::runtime_error("Invalid budget backend JSON: " + error);
  return result;
}

struct Socket {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  ~Socket() { if (fd >= 0) close(fd); }
};
}

BudgetViewerClient::BudgetViewerClient(int port) : port_(port) {
  if (port < 1 || port > 65535) throw std::runtime_error("Invalid backend port");
  const auto body = request("/api/meta");
  metadata_ = parseJson(body.data(), body.data() + body.size());
  if (metadata_["frames"].empty()) throw std::runtime_error("No saved camera views");
}

std::string BudgetViewerClient::request(const std::string& path, const std::string& body) const {
  Socket connection;
  if (connection.fd < 0) throw std::runtime_error("Cannot create budget backend socket");
  timeval timeout{30, 0};
  setsockopt(connection.fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  setsockopt(connection.fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port_);
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (connect(connection.fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0)
    throw std::runtime_error("Cannot connect to local budget backend: " + std::string(strerror(errno)));
  const std::string wire = (body.empty() ? "GET " : "POST ") + path +
      " HTTP/1.0\r\nHost: 127.0.0.1\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: " +
      std::to_string(body.size()) + "\r\n\r\n" + body;
  size_t sent = 0;
  while (sent < wire.size()) {
    const auto n = send(connection.fd, wire.data() + sent, wire.size() - sent, MSG_NOSIGNAL);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) throw std::runtime_error("Budget backend send failed");
    sent += n;
  }
  std::string response;
  char buffer[65536];
  for (;;) {
    const auto n = recv(connection.fd, buffer, sizeof(buffer), 0);
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) throw std::runtime_error("Budget backend receive failed or timed out");
    if (n == 0) break;
    response.append(buffer, n);
    if (response.size() > 256 * 1024 * 1024) throw std::runtime_error("Budget frame exceeds 256MB limit");
  }
  const auto split = response.find("\r\n\r\n");
  if (split == std::string::npos) throw std::runtime_error("Incomplete budget backend HTTP response");
  if (response.compare(0, 12, "HTTP/1.0 200") != 0 && response.compare(0, 12, "HTTP/1.1 200") != 0)
    throw std::runtime_error("Budget backend: " + response.substr(split + 4));
  const auto length = response.find("Content-Length:");
  if (length == std::string::npos ||
      std::stoull(response.substr(length + 15)) != response.size() - split - 4)
    throw std::runtime_error("Incomplete budget backend body");
  return response.substr(split + 4);
}

BudgetViewerClient::Frame BudgetViewerClient::render(const Json::Value& input) const {
  Json::StreamWriterBuilder writer;
  const auto body = request("/api/render-native", Json::writeString(writer, input));
  if (body.size() < 4) throw std::runtime_error("Missing native frame header");
  const auto* bytes = reinterpret_cast<const unsigned char*>(body.data());
  const size_t header_length = size_t(bytes[0]) | (size_t(bytes[1]) << 8) |
      (size_t(bytes[2]) << 16) | (size_t(bytes[3]) << 24);
  if (header_length > body.size() - 4) throw std::runtime_error("Truncated native frame header");
  Frame frame;
  frame.metadata = parseJson(body.data() + 4, body.data() + 4 + header_length);
  const int width = frame.metadata["width"].asInt(), height = frame.metadata["height"].asInt();
  const size_t image_bytes = size_t(width) * size_t(height) * 3 * sizeof(float);
  const size_t reference_bytes = frame.metadata["reference_bytes"].asUInt64();
  if (width < 1 || height < 1 || image_bytes != frame.metadata["image_bytes"].asUInt64() ||
      (reference_bytes != 0 && reference_bytes != image_bytes) ||
      body.size() != 4 + header_length + image_bytes + reference_bytes)
    throw std::runtime_error("Invalid native frame layout");
  frame.image.create(height, width, CV_32FC3);
  const auto* pixels = body.data() + 4 + header_length;
  std::memcpy(frame.image.data, pixels, image_bytes);
  if (reference_bytes) {
    frame.reference.create(height, width, CV_32FC3);
    std::memcpy(frame.reference.data, pixels + image_bytes, reference_bytes);
  }
  return frame;
}
