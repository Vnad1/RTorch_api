# RTorch_api — 项目状态 (authoritative)

> 这份文件是 RTorch_api 的持久真相来源。继续之前先读它。
> 每次改动手动更新"已完成/待做"段。不依赖对话记忆。

## 定位(用户原话,权威,勿改)

> 用户决策原句(来自会话日志):
> - "这个api.dll是**更扩展**的，它会**检测RTorch.exe**，然后提供**最全面的接口**，但是规则要**有人写代码添加规则**，但是原有的RTorch依然**只负责计算**，理念不变但是**社区更自由了**"
> - "这个dll复杂点，Rust提供**安全向量**，C++提供**性能**，Python**粘合代码**，C#提供**巨量引用**"
> - "反正这个**不影响RTorch基本设计**就完事，它就是一个**插件**，最后为**海量的社区**服务"

跨语言插件层,架在 **RTorch** 之上。RTorch 本身 = 通用计算框架/计算层
(不是模型库/不是 Transformer 框架),**只负责计算**。Striker = 模型框架,在 RTorch
之上,**本项目不涉及 Striker**(它在另一个对话,不用了解)。

RTorch_api**检测 RTorch.exe、提供最全面接口**,规则由社区**人写代码添加**。
对外暴露一条**稳定 C ABI** (`include/rtorch_api.h`) 给四语言社区:

| 语言 | 分工 |
|---|---|
| Rust | 安全向量/内存安全壳 (crate-type = rlib + cdylib) |
| C++ | 性能热路径 |
| Python | 粘合代码 |
| C# | 巨量引用高层 API |

**DLL 命名(已拍板)**: 用户定 **`RTorch_for_models_api.dll`**。需把 Cargo.toml
`[lib] name` 从 `rtorch_api` 改为 `rtorch_for_models_api`,`description`/Cargo 注释同步。
注: crate `package.name` 仍可叫 `rtorch_api`(crate 名),但**产出的 cdylib DLL 名**
由 `[lib] name` 决定,须为 `rtorch_for_models_api`。

**规则表作用域(已拍板)**: 用户选**全局 `Mutex<HashMap<String, RuleFn>>`**(按原始设计)。
需把 M2 的 `thread_local!` 改回全局 `static Mutex<HashMap>`,跨线程可见。
约束: `Mutex` 要求 `RuleFn` 捕获体 `Send`;裸 userdata 指针非 Send →
包装时用 `unsafe` 包 `Send`(或规则存 `(cf,userdata)` 而非闭包捕获 userdata)。

**"检测 RTorch.exe"(已拍板)**: 用户原话"它的**计算依赖 RTorch**"。含义 =
**crate 级复用 RTorch 公开 API**(其计算能力由 RTorch 提供),**不 shell 到 rtorch.exe 进程**。
当前实现(use `rtorch::formula`/`rtw`/`device`)正确,不需改。
"检测"字面应理解为"依赖/通过 RTorch 计算",非运行时探测 exe——维持现状。

## M3 落地事实(rtw.rs 有完整支撑,非 stub)

`rtorch::rtw` 公开类型/函数(M3 访问器直接接,不用重头设计):
- `Model{ name:String, version:u32, params:Vec<NamedTensor>, opt:Option<OptState> }`
- `NamedTensor{ name:String, shape:Vec<u32>, dtype:u8, data:Vec<f32> }`
- `OptState{ m:Vec<Vec<f32>>, v:Vec<Vec<f32>>, t:u64 }`
- `Memory{ fragments:Vec<MemoryFragment> }`;`MemoryFragment{ id:u64, state:Vec<f32>, strength:f32 }`
- `encode_model(&Model)->Vec<u8>` / `decode_model(&[u8])->io::Result<Model>`
- `decode`(容器)产 `Rtw{kind,dtype,shape,data,kernel}`;KIND_MODEL=2 / KIND_MEMORY=3
- model/memory 容器 dtype=DTYPE_BYTES(5), shape=params/fragments 计数

## 设计依据(原始设计,来自会话 turn 88/89,未落盘——见 DESIGN.md)

- 定位: RTorch 之上跨语言扩展层;RTorch 核心只计算;RTorch_api 提供"更全面接口 + 可插拔规则"。
- 分层: C#(巨量引用/丰富API) + Python(粘合) → 高层绑定;Rust(安全壳/cdylib) + C++(性能热路径) → 插件本体;一条稳定 C ABI(`rtorch_api.h`) → 四语言互通桥梁;RTorch crate(只计算) → 底座。
- 接口按能力分组(见 `include/rtorch_api.h`): tensor / formula session / RTW 容器 / model+memory 净荷访问器 / 规则注册。
- 里程碑: M1 C ABI头+Rust壳(张量/会话/rtw)+冒烟 ✅ → M2 公式持久会话+规则注册 ✅ → M3 model/memory 净荷访问器+RTW 完整读写 ⏳ → M4 C++薄包装 → M5 Python绑定 → M6 C# P/Invoke。
- 目录(原始设计): `src/`(lib.rs 安全壳 / ffi.rs C ABI 出口 / rule.rs / err.rs / tensor.rs / session.rs / rtw.rs) + `cpp/`(C++ 性能) + `include/rtorch_api.h` + `python/` + `csharp/` + `tests/`。**目前只有 src/{lib,ffi,rule,err}.rs + include + tests 落地;tensor.rs/session.rs/rtw.rs/err 拆分、cpp/python/csharp 目录未建。**

## 铁律

1. **绝不修改 rtorch 源码**。只复用其公开 API:
   `rtorch::formula::{run, find_compiler}`, `rtorch::rtw`, `rtorch::tensor`, `rtorch::device`, `rtorch::ops`, `rtorch::autograd`, `rtorch::gvar`。
   (注: "检测 RTorch.exe" 的具体含义——是 crate 级复用还是 shell 调用 rtorch.exe——待用户澄清。当前实现走 crate 路径。)
2. C ABI 约定: `extern "C"`, `int rc` 返回码 + `rtorch_api_last_error()`;不抛异常。
3. 内存: 每个 `*_new`/`*_decode` 配 `*_free`;Rust 壳内做边界检查。
4. 规则注册表 thread-local;Rust 闭包 `RuleFn` 不带 Send+Sync(裸 userdata 指针)。
5. 不 panic 跨 C 边界: `ffi_guard`/`ffi_guard_ptr` 抓 panic → 返回错误码/null。

## License 注意

RTorch = **LGPL-3.0**。RTorch_api 若只经公开头/API 交互(不闭静态重链接库部分),
可按插件自身许可;用户公式只含 `rtorch.h` 则非衍生作品。许可证策略未定,待用户拍板。

## 仓库 / 路径

- Git: `github.com/Vnad1/RTorch_api`(私有,已 clone)
- 本地: `D:\AP\RTorch_api`(rtorch 依赖 path = `..\rtorch`)
- 头: `include\rtorch_api.h`
- 源码: `src\{lib,err,rule,ffi}.rs`
- 测试: `tests\ffi_smoke.rs`

## 错误码

```
RTORCH_API_OK=0  E_PARAM=1  E_NOMEM=2  E_RT=3  E_IO=4  E_PARSE=5
```

## 已完成

### M1 — 骨架(✅)
- C ABI 头 + err/rule/ffi/lib 编译通过。
- `cargo build --release` 产 cdylib + rlib。
- objdump 导出 33 个 `rtorch_api_*` 符号,可 LoadLibrary。
- tensor(/session/rtw/rule) 冒烟 3 passed。

### M2 — rule C-ABI 一致性 + session 持久化(✅)
- **rule C ABI 修复**: 头声明 `rtorch_api_rule_fn` 是 C 函数指针 + userdata,
  旧 ffi 用 `Box<dyn Fn>` 参数(ABI 对不上)。改为:
  - `rule.rs`: `pub type RuleCFn = unsafe extern "C" fn(*const Blob, usize, *mut Blob, *mut c_void)`;
    新增 `register_c(name, cf, userdata)` 把 C fn 指针包成 Rust 闭包(run-time userdata
    覆盖 register 默认值)。
  - `ffi.rs`: `rtorch_api_register_rule(name, cf, userdata)` 走 `register_c`。
- **session 持久化**: 原 `session_compile` 只存 raw .cpp,`session_execute` 每次
  `formula::run(cpp)` → 每次 g++ 重编译。改为:
  - `compile_once`: `.cpp` 在 compile() 一次性编成持久 temp DLL(与 RTorch 同 flags,
    加上 `-I` 公式目录 + ref_dir,前置编译器 bin 进 PATH)。`.dll` 直接 canonicalize。
  - `OpaqueSession{dll, owns_dll}`: 缓存编译好的 DLL;`execute` 用 `formula::run(dll)`
    → 只 load/dispatch,不再重编译。`free` 清理自有 temp DLL。
  - 每 session 独立 DLL 名 `rtorch_for_models_api_rt_{pid}_{counter}`,并发不撞。
- **验证**: clippy -D 0 err;`cargo test` 4 passed;`cargo build --release` 产
  `rtorch_for_models_api.dll`,objdump 33 符号全在。

### M2.1 — 用户拍板修正(✅ 本轮)
按用户三个决策改代码:
1. **DLL 名** = `RTorch_for_models_api.dll` → Cargo.toml `[lib] name = "rtorch_for_models_api"`,
   `[package] name` 仍 `rtorch_api`(Rust 内 crate 名,测试 `use rtorch_for_models_api::...`)。
   description/Cargo 注释同步。**注**: Rust crate 名 = `[lib] name`,故测试/rust 侧
   `use` 须用 `rtorch_for_models_api`。
2. **规则表** = 全局 `Mutex<HashMap>`(按原设计,跨线程可见)。
   - `RULES: OnceLock<Mutex<HashMap<String, RuleFn>>>`(静态 Mutex 非 const,用 OnceLock)。
   - `RuleFn = Box<dyn Fn(...) -> i32 + Send>`(Send,因全局共享)。
   - `SendPtr(*mut c_void)` + `unsafe impl Send` 包装 userdata;`SendPtr::get()`
     方法读取(避免 edition 2024 disjoint capture 只捕获字段 `*mut c_void`→非 Send 的坑)。
   - `register_c` 闭包捕获 `sd: SendPtr`(Send) + `cf`(fn ptr Send)→ 闭包 Send。
3. **检测 RTorch.exe** = 计算依赖 RTorch(crate 级复用,不 shell 到 rtorch.exe)。现值正确,不改。

### M3 — model/memory 净荷访问器 + RTW 完整读写(✅ 本轮)

- `OpaqueModel{ model: rtorch::rtw::Model }`, `OpaqueMemory{ memory: rtorch::rtw::Memory }`
  (原 stub `payload: Vec<u8>` 换成真 Model/Memory 类型)。
- `model_decode` → `rtw::decode_model`;`memory_decode` → `rtw::decode_memory`(payload = RTW
  kind=2/3 的 `data` 字段)。
- 访问器全部接真字段:
  - model: name(经 `cstr_return` thread-local CString 缓冲)/ version / num_params /
    param_name / param_data(+out_numel) / param_shape_len / param_shape_dim。
  - memory: num_frags / frag_id / frag_state(+out_numel) / frag_strength。
  - 越界索引返 0/null,不 panic,无 UB。
- **`cstr_return(&str)`**: thread-local 可复用 CString 缓冲,返回借用 `const char*`,
  每次调用覆盖,不泄漏(同 last_error 模式)。
- Cargo.toml: 加 `[dev-dependencies] rtorch`(同 path),让集成测试能建真 Model/Memory payload。
- **验证**: `cargo test` 7 passed(新增 `model_accessors` / `memory_accessors` /
  `rtw_container_roundtrip`);clippy -D 0 err;`cargo build --release` 产
  `rtorch_for_models_api.dll`,objdump 33 符号全在。

### M3.5 — RTW Manifest 访问器(✅ 本轮, 跟进 RTorch 0.1.4)

RTorch 0.1.4 给 RTW 加了自描述 Manifest("来自什么库 / 在哪 / format version / 需要能力")。
RTorch_api 现在把它透给四语言:

- `OpaqueRtw` 加 `manifest: Option<rtorch::rtw::Manifest>`;`rtw_decode` 取 `rtw.manifest`。
- 新 C ABI 访问器(全部 extern "C",经 `cstr_return` 返回借用 `const char*`):
  - `rtorch_api_rtw_has_manifest(r) -> i32`
  - `..._manifest_artifact_id(r)` / `..._manifest_location(r)` / `..._manifest_format_version(r)`
  - `..._manifest_requires_count(r)` / `..._manifest_requires_dim(r, i)`
- header `include/rtorch_api.h` 同步声明;四语言绑定全加:
  - cpp `Rtw::{has_manifest,artifact_id,location,format_version,requires_count,requires_dim}`
  - python `Rtw.{has_manifest,artifact_id,location,format_version,requires}`(property/list)
  - csharp `Rtw.{HasManifest,ArtifactId,Location,FormatVersion,RequiresCount,RequiresDim}`
- **验证**: `cargo test` 22 passed(新增 `rtw_manifest_accessors`);clippy -D 干净;
  `cargo build --release` 产 dll,**objdump 39 符号**(33 + 6 manifest);header C 编译过;
  C++/Python/C# 冒烟全过。
- **测试用例注意**: RTorch_api 依赖 rtorch 的 `rtw::Rtw{}` 字面量需补 `manifest: None`
  (若 rtorch 再改该 struct,RTorch_api 要同步)。

## 待做

- **M4**: C++ 薄包装(性能热路径) — ✅ 完成
- **M5**: Python 绑定 — ✅ 完成
- **M6**: C# P/Invoke — ✅ 完成
- **方向(已拍板)**: 计算依赖 RTorch(crate),不是 shell 进程。

### M4 — C++ 薄包装(✅)
- `cpp/rtorch_for_models_api.hpp`(header-only RAII): `Tensor`/`Session`/`Rtw`/`Model`/
  `Memory` 类 + `register_rule`/`run_rule`/`last_error`;每条 C ABI 错误返 C++ `std::runtime_error`。
- `tests/cpp_smoke.cpp`: 编译+链接+运行。`g++ -std=c++17 -I include -I cpp
  tests/cpp_smoke.cpp target/release/rtorch_for_models_api.dll`(直接链 DLL)。通过。

### M5 — Python 绑定(✅)
- `python/rtorch_for_models_api.py`(ctypes): 自动定位 DLL(`RTORCH_API_DLL` env 或常见路径);
  全 33 函数签名映射;类封装 `Tensor`/`Session`/`Rtw`/`Model`/`Memory`;错误抛
  `RTorchApiError`(非零 rc + last_error);规则用 `ctypes.CFUNCTYPE` 回调。
- `tests/python_smoke.py`: tensor(含 F64 拒绝)+ rule。通过。

### M6 — C# P/Invoke(✅)
- `csharp/RTorchForModelsApi.cs`: `[DllImport("rtorch_for_models_api", CallingConvention=Cdecl)]`
  全 33 函数 + 托管类封装(disposable,`IDisposable` 配对 *_free)+ `RTorchApiException`。
  `Blob` 用 `StructLayout(Sequential)`,规则回调 `RuleFn` 委托用 IntPtr(避 marshaler 把
  数组误当单结构指针)。
- `csharp/RtApiSmoke/`: net10.0 控制台冒烟,`dotnet build` + 复制 DLL 运行。通过。

### 四语言验证(全过)
- Rust: `cargo test` 10 passed
- C++: tensor+rule OK
- Python: tensor+rule OK
- C#: tensor+rule OK
- 均链同一 `rtorch_for_models_api.dll`,33 符号 C ABI。
- 新文件: `.gitignore`(target/bin/obj/pycache/dll 不入库)、`README.md`(四语言构建运行)。

## 审计(工具排查 M1-M3, 已做, 修 2 真 bug)

### 排查手段
- 静态: `cargo clippy --all-targets -- -D warnings` 0 err(依赖 rtorch 的 18 条 warning 是 dep 的,
  非本 crate;本 crate 编译零 warning)。
- ABI: 提取 header 33 个声明 vs ffi.rs 33 个 `extern "C"` fn,一一对照片刻。
- C 可消费性: 写 `tests/header_c_check.c` 引用全部函数,`gcc -std=c11 -Wall -Wextra -c` 通过
  (证明 C ABI 桥对 C 侧也干净)。
- 动态: `cargo test` 全过(现在 10 个)。
- 内存: grep 全 unsafe/into_raw/from_raw/forget/CString 逐一审。

### 找到并修复的 2 个真 bug
1. **`rtorch_api_rtw_bytes` 内存泄漏 + ABI 契约破坏**(上一轮审计已修)。
2. **`rtorch_api_tensor_new` dtype 谎言(内存安全)**(上一轮审计已修)。

## 二轮审计(工具复查 M1-M6, 又修 2 真 bug)

### 新增手段
- **真实外部载入测试** `tests/external_dll.rs`: 用 `LoadLibrary`/`GetProcAddress` 把 DLL 当*外来*
  库解析 C ABI 符号(与 Python ctypes / C# P/Invoke 完全一致),证 DLL 真能从外部载入调用。
- **边界/panic 探针** `tests/probe.rs`: 每条 FFI 喂敌意输入(null/越界/溢出/垃圾字节),
  用 `catch_unwind` 断言不 panic、不 OOB、不跨边界崩溃。

### 找到并修复的 2 个真 bug
1. **`rtorch_api_tensor_new` numel 溢出 → 巨量分配/**abort**(鲁棒性/DoS)**:
   - 旧实现 `let numel: usize = dims.iter().product()`;敌意 dims(如 `[usize::MAX/2, usize::MAX/2]`)
     乘积溢出成 ~42.9e9 元素 → `vec![0.0f32; numel]` 试图分配 171GB → **进程 abort**(非优雅错误)。
   - 修: `checked_mul` 逐维乘,溢出返 E_PARAM;外加 `numel > (1<<29)` 上限返 E_NOMEM。
   - 附带修: 0-rank 张量 numel 原算成 1,应为 0(空张量)。
2. **`rtorch_api_memory_decode` 垃圾 payload → 巨量分配/abort**(依赖层可修,已去 rtorch 根治):
   - 根因在 rtorch `rtw::decode_model`/`decode_memory` 读 count 后 `Vec::with_capacity(count)` 无上界;
     敌意 8 字节 `[0xFF;8]` → count=0xFFFFFFFF → 尝试 171GB → abort。
   - **已在 rtorch 根治**(`D:\AP\rtorch\src\rtw.rs`): 新增 `cap_for(count, min_entry, remaining)`
     辅助——容量 clamp 到 `remaining/min_entry`,防巨量分配;
     应用到 `decode_model`(nparams/opt read_lists)、`decode_memory`(frags)、`decode`(shape rank)。
   - rtorch 加 2 回归测试:`hostile_count_does_not_abort` / `truncated_memory_returns_err`。
   - RTorch_api 的 `reject_oversized_count` 闸**保留**作纵深防御(不改 rtorch 时也挡)。
   - **注**: rtorch `build.rs:84` 有一个**早已存在**(非本次)的 clippy `collapsible_if` lint,
     与本修复无关,未动(避免影响构建可复现)。

### 二轮审计后测试(18 全过)
- ffi_smoke 10 (M1-M3 功能) + external_dll 2 (外部载入) + probe 6 (边界/panic)
- 四语言链路复跑: Rust/C++/Python/C# 全过
- rtorch 全量 test suite 过(含 rtw 11 个,新增 2 个回归)
- 产物: `rtorch_for_models_api.dll` 33 符号

### 二轮审计后测试(18 全过)
- ffi_smoke 10 (M1-M3 功能) + external_dll 2 (外部载入) + probe 6 (边界/panic)
- 四语言链路复跑: Rust/C++/Python/C# 全过
- 产物: `rtorch_for_models_api.dll` 33 符号

## 审计后测试(10 全过)
- tensor_roundtrip / tensor_dtype_contract(拒 F64/I32/null dims)
- session_compile_execute(真公式编译+执行) / session_execute_null_guard(null→E_PARAM)
- rule_register_and_run / rule_register_c_abi(真 `extern "C"` 回调+userdata) / rule_not_found
- model_accessors / memory_accessors(真 payload) / rtw_container_roundtrip(含小缓冲路径)

## 三轮审计(代码逐行 + release 编译后行为, 又修 1 真 bug)

### 发现并修复的真 bug
**`panic = "abort"` 使 release 下 catch_unwind 失效 — 安全壳契约被静默破坏**:
- `Cargo.toml` release profile 用 `panic = "abort"`,但 `ffi_guard`/`ffi_guard_ptr`
  全靠 `catch_unwind` 把内部 panic 转成 C ABI 错误码(不泄 C 边界)。
  `panic = "abort"` 在 release 下禁用 unwinding → `catch_unwind` 成 no-op,
  内部 panic(如 OOM/越界)会直接 `abort()` 杀掉整个宿主进程,契约形同虚设。
- **修**: release profile `panic = "unwind"`,恢复 catch_unwind 生效。
- **验证**: 加 lib 单元测试(`ffi_guard_catches_internal_panic` /
  `ffi_guard_ptr_returns_null_on_panic` / `ffi_guard_ok_returns_code`),
  **`cargo test --release --lib` 3 passed** —— 证明 release 下 panic 真被
  catch_unwind 抓住并返回 E_RT(若仍是 abort,测试进程会 abort,即回归触发)。

### 三轮审计核对(通过)
- **API 用法**: `rtorch::formula::run(&Path, &[], &inputs, device)` 签名/语义正确;
  `rtorch::rtw::{decode_model, decode_memory}` 用法正确;`find_compiler` 正确。
  rtw_bytes 5 语言签名一致(修过泄漏后)。`compile_once` 复刻 rtorch 编译 flags(漂移风险=技术债)。
- **跨语言 ABI**: `Blob` 布局 Rust/Python/C#/C++/header 一致
  (`{void* data; size_t len}`, x64 下 8+8, repr(C)/Sequential);
  C# 全 33 个 DllImport + RuleFn 委托均 `CallingConvention.Cdecl`(匹配 extern "C")。
- **边界/敌意输入**: probe.rs 6 项覆盖 null/越界/溢出/垃圾字节,全部不 panic 不 OOB。
- **release 编译后**: `cargo test --release` 全过(21: 3 lib + 2 external + 10 ffi_smoke + 6 probe);
  四语言复跑 Rust/C++/Python/C# 全过。

### 技术债(记录未改)
- `compile_once` 在 RTorch_api 内复刻 rtorch 的 `compile_formula`(flags/-I 硬编码),
  若 rtorch 改编译参数会漂移。可考虑 rtorch 暴露一个 `compile` 公开 API 复用。
- C# `rtorch_api_session_execute` 用单元素 `Blob[]` 传 `blob* out`(blittable 数组
  首元素=指针,能工作但易误读);建议改 `ref Blob` 更清晰。

## 构建/测试命令

```sh
cd D:\AP\RTorch_api
cargo build --release          # 产 cdylib + rlib
cargo clippy --all-targets -- -D warnings
cargo test                     # ffi_smoke (10)
# C++
g++ -std=c++17 -I include -I cpp tests/cpp_smoke.cpp target/release/rtorch_for_models_api.dll -o tests/cpp_smoke.exe
PATH="target/release;$PATH" tests/cpp_smoke.exe
# Python
PATH="target/release;$PATH" python tests/python_smoke.py
# C#
dotnet build csharp/RtApiSmoke/RtApiSmoke.csproj -c Release
```

## 环境要点

- smoke test 的 session 用例需要 g++(C:\Strawberry\c\bin) + `../rtorch/examples/formula_trig.cpp` 存在。
- git credentials 能 push rtorch + 能 clone 私有 RTorch_api。
- crate 无第三方依赖(仅 rtorch path);网络受限(crates.io 403)。
