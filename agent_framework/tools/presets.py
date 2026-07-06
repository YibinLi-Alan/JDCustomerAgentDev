"""预设装配 —— 一行拿到装好全部 11 个工具的 ``ToolRegistry``。

上层(CLI / demo / 测试)用 :func:`default_registry` 即可,不必逐个 import 工具;
要增删工具时改这里(或拿到 registry 后继续 ``register``),核心循环不动。
"""

from __future__ import annotations

from agent_framework.tools.common import create_common_tools
from agent_framework.tools.jd_mock import create_jd_tools
from agent_framework.tools.jd_mock_data import JDMockStore
from agent_framework.tools.registry import ToolRegistry


def default_registry(store: JDMockStore | None = None) -> ToolRegistry:
    """构造默认工具注册中心:8 个京东业务工具 + 3 个通用工具。

    Args:
        store: 业务工具共用的 mock 数据源;缺省用进程单例 ``DEFAULT_STORE``
            (跨工具状态变化可见),测试请注入独立的 ``JDMockStore()``。

    Returns:
        装配完成的 :class:`ToolRegistry`(11 个工具)。
    """
    return ToolRegistry([*create_jd_tools(store), *create_common_tools()])
