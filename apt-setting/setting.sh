# 1. 更换源（使用 TUNA 镜像）
sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list
sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

# 2. 安装 apt HTTPS 支持、证书、GPG 等基本工具（可能第一次 apt-get update 还会警告，忽略即可）
apt-get update --allow-insecure-repositories || true
apt-get install -y --no-install-recommends \
    apt-transport-https \
    ca-certificates \
    gnupg \
    curl \
    software-properties-common

# 3. 导入 Ubuntu 公钥（使用 gpg 下载+导入方式，稳定可靠）
gpg --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32
gpg --export --armor 3B4FE6ACC0B21F32 | gpg --dearmor -o /etc/apt/trusted.gpg.d/ubuntu.gpg

# 4. 重新更新
apt-get update
