
import asyncio
import os
from dotenv import load_dotenv
from vercel import blob
from vercel.api import session

load_dotenv()
 
async def resolve_credentials() -> blob.BlobCredentials:
    return blob.BlobCredentials(
        token=os.environ["BLOB_READ_WRITE_TOKEN"],
        store_id=os.environ["BLOB_STORE_ID_STORE_ID"],
        kind=blob.CredentialKind.OIDC
    )
 
 
async def main() -> None:
    options = blob.BlobServiceOptions(credentials_factory=resolve_credentials)
 
    async with session(service_options=[options]):
        metadata = await blob.put_blob(r"output\corona_virus.md", b"ashoka tha king", access = "public", allow_overwrite=True)
        print(metadata)
 
 
asyncio.run(main())