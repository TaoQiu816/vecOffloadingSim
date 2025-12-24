import sys
import torch


def check_environment():
    print(f"{'=' * 30}\n环境自检开始\n{'=' * 30}")

    # 1. Python 版本
    print(f"[Python]: {sys.version.split()[0]}")

    # 2. PyTorch 检测
    try:
        print(f"[PyTorch]: {torch.__version__}")
        if torch.cuda.is_available():
            print(f" -> Device: GPU ({torch.cuda.get_device_name(0)})")
        elif torch.backends.mps.is_available():
            print(f" -> Device: Mac MPS (Metal Performance Shaders)")
        else:
            print(f" -> Device: CPU")
    except ImportError:
        print("❌ PyTorch 未安装!")
        return

    # 3. PyG 检测 (最容易崩的地方)
    try:
        import torch_geometric
        from torch_geometric.nn import GATConv
        print(f"[PyG]: {torch_geometric.__version__}")

        # 简单测试 GAT 是否能运行
        conv = GATConv(16, 32, heads=2)
        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        out = conv(x, edge_index)
        print(" -> GATConv 运算测试: ✅ 通过")
    except ImportError as e:
        print(f"❌ PyG 依赖缺失: {e}")
        print("提示: 请检查 torch_scatter/sparse 是否安装正确")
        return
    except Exception as e:
        print(f"❌ PyG 运算失败: {e}")
        return

    # 4. 其它库
    libs = ['numpy', 'gym', 'networkx']
    for lib in libs:
        try:
            __import__(lib)
            print(f"[{lib}]: ✅")
        except ImportError:
            print(f"[{lib}]: ❌ 未安装")

    # 5. daggen 特殊检测
    try:
        import daggen
        print(f"[daggen]: ✅")
    except ImportError:
        print(f"[daggen]: ⚠️ 未安装 (Windows上可能需要用预生成数据)")

    print(f"\n{'=' * 30}\n环境自检完成，一切就绪！\n{'=' * 30}")


if __name__ == "__main__":
    check_environment()