import { useEffect, useState } from "react";
import { onAuthStateChanged, signInWithEmailAndPassword, signOut, User } from "firebase/auth";
import { BadgeCheck, CircleDollarSign, FileText, LayoutDashboard, LogOut, ShieldCheck, Users, WalletCards } from "lucide-react";
import { auth } from "./firebase";
import { api } from "./api";

type Me={uid:string;email:string;role:string};
type Account={account_id:string;uid:string;email:string;legal_name:string;phone:string;country:string;risk_profile:string;kyc_status:string;account_status:string;fund_id:string;base_currency:string};
type Holding={account_id:string;units:string;nav:string;market_value:string;net_contributions:string;gain_loss:string};
type Tx={transaction_id:string;kind:string;amount:string;units:string;nav:string;status:string;created_at:any};
type AdminUser={uid:string;email:string;display_name:string;disabled:boolean;email_verified:boolean;role:string;last_sign_in_at?:number};

function money(v:any,c="USD"){return new Intl.NumberFormat("en-US",{style:"currency",currency:c,maximumFractionDigits:2}).format(Number(v||0))}
function date(v:any){if(!v)return "—";if(typeof v==="string")return new Date(v).toLocaleString();const s=v._seconds||v.seconds;return s?new Date(s*1000).toLocaleString():"—"}

function Login(){
  const[email,setEmail]=useState("");const[password,setPassword]=useState("");const[err,setErr]=useState("");
  async function submit(e:React.FormEvent){e.preventDefault();setErr("");try{await signInWithEmailAndPassword(auth,email,password)}catch(x){setErr(x instanceof Error?x.message:"Sign in failed")}}
  return <div className="auth-shell"><div className="brand"><b>Q</b><div><strong>Quant Fund</strong><span>Investor portal</span></div></div>
    <form className="login-card" onSubmit={submit}><span className="eyebrow">SECURE ACCESS</span><h1>Welcome back</h1><p>View your investment account, units, transactions and statements.</p>
      <label>Email<input type="email" value={email} onChange={e=>setEmail(e.target.value)} required/></label>
      <label>Password<input type="password" value={password} onChange={e=>setPassword(e.target.value)} required/></label>
      {err&&<div className="error">{err}</div>}<button className="primary">Sign in</button><small>Firebase Authentication + server-side access control.</small>
    </form></div>
}

export default function App(){
 const[user,setUser]=useState<User|null>(null),[ready,setReady]=useState(false),[me,setMe]=useState<Me|null>(null);
 const[accounts,setAccounts]=useState<Account[]>([]),[account,setAccount]=useState<Account|null>(null),[holding,setHolding]=useState<Holding|null>(null),[txs,setTxs]=useState<Tx[]>([]);
 const[page,setPage]=useState("overview"),[err,setErr]=useState("");
 useEffect(()=>onAuthStateChanged(auth,u=>{setUser(u);setReady(true);if(!u){setMe(null);setAccounts([]);setAccount(null)}}),[]);
 async function loadAccounts(){const a=await api<Account[]>("/api/accounts");setAccounts(a);if(!account&&a.length)setAccount(a[0])}
 useEffect(()=>{if(!user)return;(async()=>{try{setMe(await api<Me>("/api/me"));await loadAccounts()}catch(e){setErr(String(e))}})()},[user]);
 async function refresh(){if(!account)return;const h=await api<Holding>("/api/accounts/"+account.account_id+"/holdings");const t=await api<Tx[]>("/api/accounts/"+account.account_id+"/transactions");setHolding(h);setTxs(t)}
 useEffect(()=>{if(account)refresh().catch(e=>setErr(String(e)))},[account]);
 if(!ready)return <div className="loading">Loading secure portal…</div>;if(!user)return <Login/>;
 const admin=me?.role==="admin";
 const nav=[["overview","Overview"],["account","My account"],["cash","Add / withdraw"],["statement","Statements"],["security","Security"]].concat(admin?[["admin","Fund operations"]]:[]);
 return <div className="shell">
  <aside><div className="brand small"><b>Q</b><div><strong>Quant Fund</strong><span>Capital portal</span></div></div>
   <label className="switcher">Account<select value={account?.account_id||""} onChange={e=>setAccount(accounts.find(a=>a.account_id===e.target.value)||null)}>{accounts.map(a=><option value={a.account_id} key={a.account_id}>{a.legal_name||a.account_id}</option>)}</select></label>
   <nav>{nav.map(n=><button className={page===n[0]?"active":""} onClick={()=>setPage(n[0])} key={n[0]}>{n[0]==="overview"?<LayoutDashboard/>:n[0]==="account"?<WalletCards/>:n[0]==="cash"?<CircleDollarSign/>:n[0]==="statement"?<FileText/>:n[0]==="admin"?<Users/>:<ShieldCheck/>}<span>{n[1]}</span></button>)}</nav>
   <div className="side-bottom"><div className="identity"><b>{me?.email}</b><span>{me?.role}</span></div><button onClick={()=>signOut(auth)}><LogOut/>Sign out</button></div>
  </aside>
  <main><header><div><span className="eyebrow">INVESTOR PORTAL</span><h1>{title(page)}</h1></div><div className="status"><span className={"pill "+(account?.kyc_status==="verified"?"ok":"warn")}><BadgeCheck/>{account?.kyc_status||"No account"}</span></div></header>
   {err&&<div className="error banner">{err}</div>}
   {!account&&<CreateAccount done={loadAccounts}/>}
   {account&&page==="overview"&&<Overview a={account} h={holding} tx={txs}/>}
   {account&&page==="account"&&<AccountPanel a={account} h={holding}/>}
   {account&&page==="cash"&&<Cash a={account} h={holding} done={refresh}/>}
   {account&&page==="statement"&&<Statement a={account} h={holding} tx={txs}/>}
   {account&&page==="security"&&<Security a={account} me={me}/>}
   {admin&&page==="admin"&&<Admin/>}
  </main>
 </div>
}

function title(p:string){return ({overview:"Portfolio overview",account:"Investment account",cash:"Add or withdraw",statement:"Statements & activity",security:"Account security",admin:"Fund operations"} as any)[p]||"Portal"}
function FlowBars({tx}:{tx:Tx[]}){
 const rows=[...tx].reverse().filter(t=>t.status==="posted");
 let running=0;const vals=rows.map(t=>{const a=Number(t.amount||0);running+=t.kind==="subscription"?a:t.kind==="redemption"?-a:0;return Math.max(running,0)});
 const max=Math.max(...vals,1);if(!vals.length)return <div className="empty">Activity appears after transactions are posted.</div>;
 return <div className="bars">{vals.slice(-24).map((v,i)=><i key={i} title={money(v)} style={{height:Math.max(4,v/max*100)+"%"}}/>)}</div>
}
function Metric(p:{label:string;value:string;sub?:string}){return <div className="metric"><span>{p.label}</span><strong>{p.value}</strong>{p.sub&&<small>{p.sub}</small>}</div>}
function Detail(p:{label:string;value:any}){return <div className="detail"><span>{p.label}</span><b>{String(p.value||"—")}</b></div>}

function Overview({a,h,tx}:{a:Account;h:Holding|null;tx:Tx[]}){
 return <><section className="metrics"><Metric label="Portfolio value" value={money(h?.market_value,a.base_currency)}/><Metric label="Units held" value={Number(h?.units||0).toLocaleString()}/><Metric label="NAV / unit" value={money(h?.nav,a.base_currency)}/><Metric label="Gain / loss" value={money(h?.gain_loss,a.base_currency)}/></section>
 <section className="grid2"><div className="panel hero"><span className="eyebrow">ACCOUNT VALUE</span><h2>{money(h?.market_value,a.base_currency)}</h2><p>Your balance is represented by units multiplied by the latest published NAV.</p><FlowBars tx={tx}/></div>
 <div className="panel"><span className="eyebrow">ACCOUNT STATUS</span><h2>Investor profile</h2><Detail label="Legal name" value={a.legal_name}/><Detail label="KYC" value={a.kyc_status}/><Detail label="Status" value={a.account_status}/><Detail label="Risk profile" value={a.risk_profile}/><Detail label="Fund" value={a.fund_id}/></div></section>
 <div className="panel"><div className="panel-head"><div><span className="eyebrow">RECENT ACTIVITY</span><h2>Transactions</h2></div></div><TxTable rows={tx.slice(0,8)} currency={a.base_currency}/></div></>
}

function AccountPanel({a,h}:{a:Account;h:Holding|null}){return <section className="grid2"><div className="panel"><span className="eyebrow">HOLDINGS</span><h2>Unit account</h2><div className="big">{money(h?.market_value,a.base_currency)}</div><Detail label="Units" value={h?.units}/><Detail label="NAV" value={money(h?.nav,a.base_currency)}/><Detail label="Net contributions" value={money(h?.net_contributions,a.base_currency)}/><Detail label="Gain / loss" value={money(h?.gain_loss,a.base_currency)}/></div><div className="panel"><span className="eyebrow">PROFILE</span><h2>Registered details</h2><Detail label="Name" value={a.legal_name}/><Detail label="Email" value={a.email}/><Detail label="Phone" value={a.phone}/><Detail label="Country" value={a.country}/><Detail label="KYC status" value={a.kyc_status}/></div></section>}

function Cash({a,h,done}:{a:Account;h:Holding|null;done:()=>Promise<void>}){
 const[mode,setMode]=useState("add"),[amount,setAmount]=useState(""),[ref,setRef]=useState(""),[msg,setMsg]=useState("");
 async function submit(e:React.FormEvent){e.preventDefault();try{const path=mode==="add"?"/api/subscriptions":"/api/redemptions";await api(path,{method:"POST",body:JSON.stringify({account_id:a.account_id,amount:Number(amount),external_ref:ref})});setMsg(mode==="add"?"Funding request submitted for cash verification. Units post only after operations approval.":"Redemption request submitted for approval.");setAmount("");await done()}catch(x){setMsg(String(x))}}
 return <section className="grid2"><form className="panel" onSubmit={submit}><div className="segments"><button type="button" className={mode==="add"?"on":""} onClick={()=>setMode("add")}>Add funds</button><button type="button" className={mode==="withdraw"?"on":""} onClick={()=>setMode("withdraw")}>Withdraw</button></div><span className="eyebrow">{mode==="add"?"SUBSCRIPTION":"REDEMPTION"}</span><h2>{mode==="add"?"Purchase units":"Request withdrawal"}</h2><label>Amount ({a.base_currency})<input type="number" min="0" step=".01" value={amount} onChange={e=>setAmount(e.target.value)} required/></label>{mode==="add"&&<label>Payment reference<input value={ref} onChange={e=>setRef(e.target.value)}/></label>}<button className="primary" disabled={a.kyc_status!=="verified"}>{mode==="add"?"Add funds":"Request withdrawal"}</button>{a.kyc_status!=="verified"&&<div className="notice">KYC verification is required.</div>}{msg&&<div className="notice">{msg}</div>}</form><div className="panel"><span className="eyebrow">AVAILABLE</span><h2>Account liquidity</h2><div className="big">{money(h?.market_value,a.base_currency)}</div><Detail label="Units" value={h?.units}/><Detail label="NAV" value={money(h?.nav,a.base_currency)}/><p className="muted">Settlement, cut-off times and fees should be configured to match the licensed product before launch.</p></div></section>
}

function Statement({a,h,tx}:{a:Account;h:Holding|null;tx:Tx[]}){return <><section className="metrics compact"><Metric label="Statement value" value={money(h?.market_value,a.base_currency)}/><Metric label="Units" value={h?.units||"0"}/><Metric label="NAV" value={money(h?.nav,a.base_currency)}/><Metric label="Transactions" value={String(tx.length)}/></section><div className="panel"><div className="panel-head"><div><span className="eyebrow">LEDGER</span><h2>Account statement</h2></div><button onClick={()=>window.print()}>Print / save PDF</button></div><TxTable rows={tx} currency={a.base_currency}/></div></>}

function Security({a,me}:{a:Account;me:Me|null}){return <section className="grid2"><div className="panel"><span className="eyebrow">IDENTITY</span><h2>Account protection</h2><Detail label="Firebase UID" value={me?.uid}/><Detail label="Role" value={me?.role}/><Detail label="KYC" value={a.kyc_status}/><Detail label="Account state" value={a.account_status}/><div className="secure"><ShieldCheck/><div><b>Server-authorized access</b><span>Ownership and role checks are enforced by the API.</span></div></div></div><div className="panel"><span className="eyebrow">CONTROLS</span><h2>Operational safeguards</h2><Detail label="Subscriptions" value="KYC gated"/><Detail label="Redemptions" value="Approval workflow"/><Detail label="Audit" value="Hashed event trail"/><Detail label="Trading" value="Separate production risk gate"/></div></section>}

function TxTable({rows,currency}:{rows:Tx[];currency:string}){if(!rows.length)return <div className="empty">No transactions yet.</div>;return <div className="tablewrap"><table><thead><tr><th>Date</th><th>Type</th><th>Amount</th><th>Units</th><th>NAV</th><th>Status</th></tr></thead><tbody>{rows.map((t,i)=><tr key={t.transaction_id||i}><td>{date(t.created_at)}</td><td>{t.kind.split("_").join(" ")}</td><td>{money(t.amount,currency)}</td><td>{Number(t.units).toLocaleString()}</td><td>{money(t.nav,currency)}</td><td><span className={"pill "+(t.status==="posted"?"ok":"warn")}>{t.status}</span></td></tr>)}</tbody></table></div>}

function CreateAccount({done}:{done:()=>Promise<void>}){const[name,setName]=useState(""),[country,setCountry]=useState(""),[phone,setPhone]=useState("");async function submit(e:React.FormEvent){e.preventDefault();await api("/api/accounts",{method:"POST",body:JSON.stringify({legal_name:name,country,phone,risk_profile:"moderate"})});await done()}return <form className="panel narrow" onSubmit={submit}><span className="eyebrow">ONBOARDING</span><h2>Create investment account</h2><p>The account starts pending KYC.</p><label>Legal name<input value={name} onChange={e=>setName(e.target.value)} required/></label><label>Phone<input value={phone} onChange={e=>setPhone(e.target.value)}/></label><label>Country<input value={country} onChange={e=>setCountry(e.target.value)}/></label><button className="primary">Create account</button></form>}

function Admin(){
 const[users,setUsers]=useState<AdminUser[]>([]),[investors,setInvestors]=useState<Account[]>([]),[subscriptions,setSubscriptions]=useState<any[]>([]),[redemptions,setRedemptions]=useState<any[]>([]),[navHistory,setNavHistory]=useState<any[]>([]);
 const[tab,setTab]=useState("users"),[show,setShow]=useState(false),[email,setEmail]=useState(""),[password,setPassword]=useState(""),[role,setRole]=useState("investor"),[name,setName]=useState(""),[navValue,setNavValue]=useState(""),[sourceEquity,setSourceEquity]=useState(""),[navDate,setNavDate]=useState(new Date().toISOString().slice(0,10));
 async function load(){const u=await api<AdminUser[]>("/api/admin/users");const a=await api<Account[]>("/api/accounts");setUsers(u);setInvestors(a);try{setSubscriptions(await api<any[]>("/api/admin/subscriptions"))}catch{}try{setRedemptions(await api<any[]>("/api/admin/redemptions"))}catch{}try{setNavHistory(await api<any[]>("/api/fund/nav/history"))}catch{}}
 useEffect(()=>{load()},[]);
 async function changeRole(uid:string,r:string){await api("/api/admin/users/"+uid+"/role",{method:"PATCH",body:JSON.stringify({role:r})});await load()}
 async function toggle(uid:string,d:boolean){await api("/api/admin/users/"+uid+"/disabled",{method:"PATCH",body:JSON.stringify({disabled:d})});await load()}
 async function create(e:React.FormEvent){e.preventDefault();await api("/api/admin/users",{method:"POST",body:JSON.stringify({email,password,role,display_name:name})});setShow(false);setEmail("");setPassword("");await load()}
 async function kyc(id:string,status:string){await api("/api/admin/accounts/"+id+"/kyc",{method:"POST",body:JSON.stringify({status,note:"Updated from fund operations console"})});await load()}
 async function approveSubscription(id:string){await api("/api/admin/subscriptions/"+id+"/approve",{method:"POST"});await load()}
 async function approveRedemption(id:string){await api("/api/admin/redemptions/"+id+"/approve",{method:"POST"});await load()}
 async function publishNav(e:React.FormEvent){e.preventDefault();await api("/api/admin/nav",{method:"POST",body:JSON.stringify({nav_per_unit:Number(navValue),valuation_date:navDate,source_equity:Number(sourceEquity),notes:"Published from React fund operations console"})});setNavValue("");setSourceEquity("");await load()}
 return <><div className="adminbar"><div><span className="eyebrow">FUND OPERATIONS</span><h2>Investor operations</h2></div><button className="primary" onClick={()=>setShow(!show)}>+ Add user</button></div>
 <div className="segments"><button className={tab==="users"?"on":""} onClick={()=>setTab("users")}>Users</button><button className={tab==="investors"?"on":""} onClick={()=>setTab("investors")}>KYC & accounts</button><button className={tab==="subscriptions"?"on":""} onClick={()=>setTab("subscriptions")}>Subscriptions</button><button className={tab==="redemptions"?"on":""} onClick={()=>setTab("redemptions")}>Redemptions</button><button className={tab==="nav"?"on":""} onClick={()=>setTab("nav")}>NAV</button></div>
 {show&&<form className="panel formrow" onSubmit={create}><input placeholder="Display name" value={name} onChange={e=>setName(e.target.value)}/><input type="email" placeholder="Email" value={email} onChange={e=>setEmail(e.target.value)} required/><input type="password" minLength={8} placeholder="Temporary password" value={password} onChange={e=>setPassword(e.target.value)} required/><select value={role} onChange={e=>setRole(e.target.value)}>{roles().map(x=><option key={x}>{x}</option>)}</select><button className="primary">Create</button></form>}
 {tab==="users"&&<div className="panel"><div className="tablewrap"><table><thead><tr><th>User</th><th>Role</th><th>Verified</th><th>Status</th><th>Last sign in</th><th></th></tr></thead><tbody>{users.map(u=><tr key={u.uid}><td><b>{u.display_name||u.email}</b><small>{u.email}</small></td><td><select value={u.role} onChange={e=>changeRole(u.uid,e.target.value)}>{roles().map(x=><option key={x}>{x}</option>)}</select></td><td>{u.email_verified?"Yes":"No"}</td><td><span className={"pill "+(u.disabled?"danger":"ok")}>{u.disabled?"Disabled":"Active"}</span></td><td>{u.last_sign_in_at?new Date(u.last_sign_in_at).toLocaleString():"—"}</td><td><button onClick={()=>toggle(u.uid,!u.disabled)}>{u.disabled?"Enable":"Disable"}</button></td></tr>)}</tbody></table></div></div>}
 {tab==="investors"&&<div className="panel"><div className="tablewrap"><table><thead><tr><th>Investor</th><th>Account</th><th>KYC</th><th>Status</th><th>Risk profile</th><th>Operations</th></tr></thead><tbody>{investors.map(a=><tr key={a.account_id}><td><b>{a.legal_name}</b><small>{a.email}</small></td><td>{a.account_id}</td><td><span className={"pill "+(a.kyc_status==="verified"?"ok":"warn")}>{a.kyc_status}</span></td><td>{a.account_status}</td><td>{a.risk_profile}</td><td><select value={a.kyc_status} onChange={e=>kyc(a.account_id,e.target.value)}>{["not_started","submitted","in_review","verified","rejected","expired"].map(x=><option key={x}>{x}</option>)}</select></td></tr>)}</tbody></table></div></div>}
 {tab==="subscriptions"&&<div className="panel"><div className="tablewrap"><table><thead><tr><th>Requested</th><th>Account</th><th>Amount</th><th>Payment ref</th><th>Indicative NAV</th><th></th></tr></thead><tbody>{subscriptions.map(r=><tr key={r.transaction_id||r.id}><td>{date(r.created_at)}</td><td>{r.account_id}</td><td>{money(r.amount,r.currency||"USD")}</td><td>{r.external_ref||"—"}</td><td>{money(r.nav,r.currency||"USD")}</td><td><button className="primary" onClick={()=>approveSubscription(r.transaction_id||r.id)}>Confirm cash & post units</button></td></tr>)}</tbody></table></div>{!subscriptions.length&&<div className="empty">No pending subscriptions.</div>}</div>}\n {tab==="redemptions"&&<div className="panel"><div className="tablewrap"><table><thead><tr><th>Requested</th><th>Account</th><th>Amount</th><th>Units</th><th>NAV</th><th></th></tr></thead><tbody>{redemptions.map(r=><tr key={r.transaction_id||r.id}><td>{date(r.created_at)}</td><td>{r.account_id}</td><td>{money(r.amount,r.currency||"USD")}</td><td>{r.units}</td><td>{money(r.nav,r.currency||"USD")}</td><td><button className="primary" onClick={()=>approveRedemption(r.transaction_id||r.id)}>Approve</button></td></tr>)}</tbody></table></div>{!redemptions.length&&<div className="empty">No pending redemptions.</div>}</div>}
 {tab==="nav"&&<section className="grid2"><form className="panel" onSubmit={publishNav}><span className="eyebrow">VALUATION</span><h2>Publish NAV</h2><label>Valuation date<input type="date" value={navDate} onChange={e=>setNavDate(e.target.value)} required/></label><label>NAV per unit<input type="number" step=".00000001" min="0" value={navValue} onChange={e=>setNavValue(e.target.value)} required/></label><label>Source portfolio equity<input type="number" step=".01" min="0" value={sourceEquity} onChange={e=>setSourceEquity(e.target.value)} required/></label><button className="primary">Publish NAV</button><p className="muted">Only publish after valuation, reconciliation and custodian/operations checks.</p></form><div className="panel"><span className="eyebrow">NAV HISTORY</span><h2>Published valuations</h2><div className="tablewrap"><table><thead><tr><th>Date</th><th>NAV</th><th>Source equity</th></tr></thead><tbody>{navHistory.slice(-30).reverse().map(n=><tr key={n.id||n.valuation_date}><td>{n.valuation_date}</td><td>{money(n.nav_per_unit)}</td><td>{money(n.source_equity)}</td></tr>)}</tbody></table></div></div></section>}
 </> 
}
function roles(){return["investor","viewer","trader","risk_approver","execution_approver","admin"]}
