import logging
import tenacity

class DbWorker:
    def __init__(self, remote_services):
        self.remote_services = remote_services

    async def read_messages(self,
                            table_name: str,
                            date_from: str,
                            date_to: str,
                            fields=["offer_id", "payload", "time"],
                            limit: int = 0,
                            random_sample: bool = False
                            ):
        query = f"SELECT {', '.join(fields)} FROM {table_name} WHERE time BETWEEN '{str(date_from)}' AND '{str(date_to)}'"
        if random_sample or limit > 0:
            logging.info('Fetching a random sample.')
            query += ' ORDER BY RAND()'
        if limit > 0:
            logging.info(f'Fetching {limit} rows.')
            query += f" LIMIT {limit}"

        res = await self.execute_query(query)
        return res

    @tenacity.retry(
        reraise=True,
        stop=tenacity.stop_after_attempt(10),
        wait=tenacity.wait_random(min=0, max=2))
    async def execute_query(self, query: str):
        # returns the output of `query` against Dexter db as tuple of tuples
        async with self.remote_services.get("dexter_db").use_master() as master_db, master_db.cursor() as cursor:
            await cursor.execute(query)
            res = await cursor.fetchall()
            return res
