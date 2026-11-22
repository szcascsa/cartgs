"""冻结后新视角：58%连续两帧、23%/81%跳转；不读取质量后调整采样。"""
import sys
import release_joint
import release_thin_control


def unseen_motions(source, quick=False):
    n = len(source.scene.views)
    result = [('continuous', int(.58 * n)), ('continuous', int(.58 * n) + 1),
              ('jump', int(.23 * n)), ('jump', int(.81 * n))]
    original = {frame for _, frame in release_joint.motions_original(source)}
    assert len({frame for _, frame in result}) == 4
    assert all(0 <= frame < n and frame not in original for _, frame in result)
    return result


if __name__ == '__main__':
    kind = sys.argv.pop(1)
    release_joint.motions_original = release_joint.motions
    if kind == 'joint':
        release_joint.motions = unseen_motions
        release_joint.main()
    elif kind == 'thin':
        release_thin_control.motions = unseen_motions
        release_thin_control.main()
    else:
        raise ValueError(kind)
