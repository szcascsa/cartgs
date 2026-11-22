"""独立服务器环境入口，统一绑定本轮源码、依赖和构建目录。"""
from common import *


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='benchmark':
        sys.argv.pop(1)
        from benchmark_compensation_student import main
    else:
        from distill_compensation import main
    main()
