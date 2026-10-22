import { app } from "../../scripts/app.js";

/**
 * XiMultiFolderImageLoader - show only the first `folder_count` folder_N widgets.
 *
 * 为什么不再从 this.widgets 里剔除控件
 * -----------------------------------
 * 前端把图序列化成 prompt 时是**实时**读取控件数组的：
 *
 *     // ExecutableNodeDTO
 *     get widgets() { return this.node.widgets }
 *
 *     // graphToPrompt()
 *     let { widgets: n } = e;
 *     for (let [r, i] of n.entries()) {
 *         if (!i.name || i.options.serialize === false) continue;
 *         t[i.name] = ...i.value;              // 只有这两种情况会被跳过
 *     }
 *
 * 所以一旦把 widget 从数组里摘掉，它的值就不会进 prompt；而 folder_4..folder_10
 * 在 object_info 里是 required，后端 execution.validate_prompt() 会直接拒绝排队：
 *
 *     Required input is missing: folder_4 ... folder_10
 *
 * 现在改用前端自己的可见性开关（widget.hidden），它只影响绘制与布局：
 *
 *     deriveWidgetVisibility(w) -> suppression.byExtension = w.hidden
 *     occupiesCanvasRow()       -> byExtension ? false   // 不占行高
 *     isWidgetVisibleOnSurface()-> byExtension ? false   // 不绘制、不进面板列表
 *
 * widget 始终留在 node.widgets 里，值照常序列化。
 */

const NODE_NAME = "XiMultiFolderImageLoader";
const FOLDER_SLOTS = 10;

/** 读取前端可见性记录（若已解析过），否则回落到 widget.hidden。 */
function setHidden(widget, hidden) {
  if (!widget) return;
  widget.hidden = hidden;
  widget.options = widget.options || {};
  widget.options.hidden = hidden;
  // 前端可能已经缓存了 visibility 对象，保持同步以免读到旧值。
  if (widget.visibility && widget.visibility.suppression) {
    widget.visibility.suppression.byExtension = hidden;
  }
}

/** folder_N 的序号；folder_count 不是端口，返回 null。 */
function slotIndex(name) {
  const m = /^folder_(\d+)$/.exec(name || "");
  return m ? parseInt(m[1], 10) : null;
}

app.registerExtension({
  name: "XiNodes.MultiFolderDynamicUI",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

      this.updateDynUI = function () {
        try {
          if (!this.widgets) return;

          const cntWidget = this.widgets.find(w => w.name === "folder_count");
          if (!cntWidget) return;

          let count = Number(cntWidget.value);
          if (!Number.isFinite(count)) count = 3;
          count = Math.min(FOLDER_SLOTS, Math.max(1, Math.round(count)));

          // 1. 控件只隐藏，不移除 —— 保证 folder_N 的值仍然进入 prompt
          for (const w of this.widgets) {
            const idx = slotIndex(w.name);
            if (idx !== null) setHidden(w, idx > count);
          }

          // 2. 输出插槽群仍按 count 伸缩（端口没有"值要序列化"的问题）
          if (this.outputs) {
            const fnIndex = this.outputs.findIndex(o => o.name === "FILENAME");
            if (fnIndex !== -1) {
              const fnOutputObj = this.outputs.splice(fnIndex, 1)[0];

              let currentCount = Math.floor(this.outputs.length / 2);
              if (currentCount < count) {
                for (let i = currentCount + 1; i <= count; i++) {
                  this.addOutput("IMAGE_" + i, "IMAGE");
                  this.addOutput("FILE_PATH_" + i, "STRING");
                }
              } else if (currentCount > count) {
                for (let i = currentCount * 2 - 1; i >= count * 2; i--) {
                  this.removeOutput(i);
                }
              }

              // FILENAME 始终留在最底部
              this.outputs.push(fnOutputObj);
            }
          }

          if (this.computeSize) {
            const size = this.computeSize();
            // 只撑大不缩小，避免覆盖用户手动拉宽的尺寸
            this.size[0] = Math.max(this.size[0], size[0]);
            this.size[1] = size[1];
          }

          this.setDirtyCanvas(true, true);
        } catch (e) {
          console.error("[XiNodes DynamicUI Error]", e);
        }
      };

      this._last_dyn_count = -1;

      // 新建/刷新后先梳理一次（默认 folder_count=3，只露出 folder_1..3）
      setTimeout(() => {
        const cntWidget = this.widgets && this.widgets.find(w => w.name === "folder_count");
        if (cntWidget) {
          const v = Number(cntWidget.value);
          this._last_dyn_count = Number.isFinite(v) ? Math.round(v) : 3;
        }
        this.updateDynUI();
      }, 100);

      return result;
    };

    // 沿用原作者的做法：用绘制心跳检测 folder_count 变化，
    // 比订阅 callback/事件更不容易被其它插件脱钩。
    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      const result = onDrawForeground ? onDrawForeground.apply(this, arguments) : undefined;

      if (this.widgets && this.updateDynUI) {
        const cntWidget = this.widgets.find(w => w.name === "folder_count");
        if (cntWidget) {
          let currentCount = Number(cntWidget.value);
          if (!Number.isFinite(currentCount)) currentCount = 3;
          currentCount = Math.round(currentCount);
          if (this._last_dyn_count !== currentCount) {
            this._last_dyn_count = currentCount;
            this.updateDynUI();
          }
        }
      }
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      if (this.updateDynUI) this.updateDynUI();
      return result;
    };
  },
});
