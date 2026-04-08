import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "XiNodes.MultiFolderDynamicUI",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "XiMultiFolderImageLoader") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                this.updateDynUI = function() {
                    try {
                        if (!this.widgets || !this.outputs) return;
                        
                        // 首次运行该函数时，在尚未发生破坏前立刻缓存所有的组件结构副本
                        if (!this._full_widgets_cache) {
                            this._full_widgets_cache = [...this.widgets];
                        }
                        
                        const cntWidget = this.widgets.find(w => w.name === "folder_count") || this._full_widgets_cache.find(w => w.name === "folder_count");
                        if (!cntWidget) return;
                        
                        let count = parseInt(cntWidget.value || 3, 10);
                        if (count < 1) count = 1;
                        if (count > 10) count = 10;
                        
                        // 1. 完全剔除隐去多余选项，通过映射全缓存内存进行切段来重新装填到展示名单
                        this.widgets = this._full_widgets_cache.filter(w => {
                            if (w.name && w.name.startsWith("folder_")) {
                                const num = parseInt(w.name.split("_")[1]);
                                if (!isNaN(num)) {
                                    return num <= count;
                                }
                            }
                            return true;
                        });

                        // 2. 动态调节 Outputs 的插槽群
                        const fnIndex = this.outputs.findIndex(o => o.name === "FILENAME");
                        if (fnIndex !== -1) {
                            const fnOutputObj = this.outputs[fnIndex];
                            
                            // 先将其暂且安全地抽取离港
                            this.outputs.splice(fnIndex, 1);
                            
                            let currentCount = Math.floor(this.outputs.length / 2);

                            // 进行原生端点伸展
                            if (currentCount < count) {
                                for (let i = currentCount + 1; i <= count; i++) {
                                    this.addOutput("IMAGE_" + i, "IMAGE");
                                    this.addOutput("FILE_PATH_" + i, "STRING");
                                }
                            } else if (currentCount > count) {
                                // 极其原生的收缩剔除与斩断残留虚线
                                for (let i = currentCount * 2 - 1; i >= count * 2; i--) {
                                    this.removeOutput(i);
                                }
                            }
                            
                            // 最终保证将唯一的 FILENAME 退还原位至最底部
                            this.outputs.push(fnOutputObj);
                        }
                        
                        if (this.computeSize) {
                             const size = this.computeSize();
                             this.size[0] = Math.max(this.size[0], size[0]);
                             this.size[1] = size[1];
                        }
                        
                        this.setDirtyCanvas(true, true);

                    } catch (e) {
                         console.error("[XiNodes DynamicUI Error]", e);
                    }
                }
                
                // 将最后执行的状态打上时间戳或值记忆
                this._last_dyn_count = -1;

                setTimeout(() => {
                     const cntWidget = this.widgets.find(w => w.name === "folder_count");
                     if (cntWidget) {
                         this._last_dyn_count = parseInt(cntWidget.value || 3, 10);
                     }
                     this.updateDynUI();
                }, 100);

                return result;
            }
            
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                const result = onDrawForeground ? onDrawForeground.apply(this, arguments) : undefined;
                
                if (this.widgets && this._full_widgets_cache) {
                    const cntWidget = this.widgets.find(w => w.name === "folder_count") || this._full_widgets_cache.find(w => w.name === "folder_count");
                    if (cntWidget) {
                        let currentCount = parseInt(cntWidget.value || 3, 10);
                        if (this._last_dyn_count !== currentCount) {
                            this._last_dyn_count = currentCount;
                            if (this.updateDynUI) {
                                this.updateDynUI();
                            }
                        }
                    }
                }
                return result;
            }
            
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                if(this.updateDynUI) {
                     this.updateDynUI();
                }
                return result;
            }
        }
    }
});
