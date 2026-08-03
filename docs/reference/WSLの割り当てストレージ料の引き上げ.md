Used tool: manage_task
Ran command: `cmd /c powershell -Command "Get-AppxPackage *Ubuntu* | Select-Object -ExpandProperty PackageFamilyName"`
Ran command: `cmd /c wsl -l -v`
Ran command: `cmd /c powershell -Command "Get-ChildItem '$env:LOCALAPPDATA\Packages\*Ubuntu*' -Recurse -Filter 'ext4.vhdx' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName"`
Ran command: `cmd /c powershell -Command "Get-ChildItem 'C:\Users\saiha\AppData\Local\wsl' -Recurse -Filter 'ext4.vhdx' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName"`

### 結論

はい、データを一切削除せずに **WSL2（Ubuntu）の最大ストレージ容量を 75 GB（またはそれ以上）に拡大拡張することが可能** です。

ご自身の環境の Ubuntu 仮想ディスクファイル（`ext4.vhdx`）の正確なパスを特定しました：
`C:\Users\saiha\AppData\Local\wsl\{410e7fcd-c2cf-498b-9baa-2b5a5c4db485}\ext4.vhdx`

以下の手順に従って、Windows 側から容量を 75 GB に拡張してください。

---

### ディスク容量拡大手順（データ保持）

#### 手順 1: WSL をシャットダウンする
Windows の **PowerShell**（またはコマンドプロンプト `cmd`）を開き、実行中の WSL を停止します。

```powershell
wsl --shutdown
```

#### 手順 2: `diskpart` ツールで仮想ディスクを 75 GB に拡張する
Windows の **コマンドプロンプト (cmd)** または **PowerShell** を開き、`diskpart` を起動して拡張を行います。

```cmd
diskpart
```
`DISKPART>` プロンプトが表示されたら、順番に以下のコマンドを入力します：

```cmd
select vdisk file="C:\Users\saiha\AppData\Local\wsl\{410e7fcd-c2cf-498b-9baa-2b5a5c4db485}\ext4.vhdx"
expand vdisk maximum=76800
exit
```
*(※ `76800` MB = 75 GB です。100 GB にしたい場合は `102400` を指定してください)*

#### 手順 3: Ubuntu 側でパーティションを認識・自動拡大させる
1. ターミナルから WSL (Ubuntu) を起動します。
   ```bash
   wsl
   ```

2. ディスク領域（ファイルシステム）を 75 GB 全域まで広げます：
   ```bash
   sudo resize2fs /dev/sdd
   ```

3. 容量が拡大されたか確認します：
   ```bash
   df -h /
   ```
   `Size` が `75G`（または指定したサイズ）に変更され、空き容量が大幅に増えていれば完了です。