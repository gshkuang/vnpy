import yaml
import importlib
import json
from typing import Dict, Any, Type
from vnpy.alpha import AlphaStrategy


class StrategyConfig:
    """策略配置管理器"""
    
    def __init__(self, config_file: str = "strategy_configs.yaml"):
        self.config_file = config_file
        self.configs = self.load_configs()
        self._strategy_classes = {}
    
    def load_configs(self) -> Dict[str, Any]:
        """加载策略配置"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    import json
                    return json.load(f)
        except FileNotFoundError:
            print(f"配置文件 {self.config_file} 不存在")
            return {}
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            print(f"配置文件格式错误: {e}")
            return {}
    
    def get_strategy_class(self, strategy_name: str) -> Type[AlphaStrategy]:
        """获取策略类"""
        if strategy_name not in self.configs:
            raise ValueError(f"策略 {strategy_name} 不存在于配置中")
        
        if strategy_name in self._strategy_classes:
            return self._strategy_classes[strategy_name]
        
        config = self.configs[strategy_name]
        module_name = config["module"]
        class_name = config["class_name"]
        
        try:
            module = importlib.import_module(module_name)
            strategy_class = getattr(module, class_name)
            self._strategy_classes[strategy_name] = strategy_class
            return strategy_class
        except (ImportError, AttributeError) as e:
            raise ImportError(f"无法导入策略 {strategy_name}: {e}")
    
    def get_default_params(self, strategy_name: str) -> Dict[str, Any]:
        """获取策略默认参数"""
        if strategy_name not in self.configs:
            raise ValueError(f"策略 {strategy_name} 不存在于配置中")
        
        return self.configs[strategy_name]["default_params"].copy()
    
    def get_param_ranges(self, strategy_name: str) -> Dict[str, list]:
        """获取策略参数范围（用于优化）
        返回格式: {param_name: [min_value, max_value, step]}
        """
        if strategy_name not in self.configs:
            raise ValueError(f"策略 {strategy_name} 不存在于配置中")
        
        return self.configs[strategy_name]["param_ranges"].copy()
    
    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """获取策略信息"""
        if strategy_name not in self.configs:
            raise ValueError(f"策略 {strategy_name} 不存在于配置中")
        
        return self.configs[strategy_name].copy()
    
    def list_strategies(self) -> list:
        """列出所有可用策略"""
        return list(self.configs.keys())